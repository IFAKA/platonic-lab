"""platonic-lab: does "good views" beat "bad views" and next-token prediction, with human captions only?

PRE-REGISTERED (before any run; do not edit after seeing data):
  Same text encoder, same captions, same steps/batch/lr, same seeds (0,1,2) for all three objectives.
  Score = mutual 10-NN between text emb (caption 0) and frozen DINOv2 emb on 1000 test images (chance .01),
          plus image->caption R@1 with one rotation (Procrustes) fitted on 1000 val pairs.
  Decision (paired by seed, final step): GOOD wins if mean(good-bad) and mean(good-ntp) in mkNN both > 0
  and each paired difference > 2x the seed spread of that difference; TIE if either gap < 0.005; else LOSE.
Usage: lab.py DATA.pt OBJ SEED [STEPS]   OBJ in good|bad|ntp  -> results/OBJ_sSEED.json
"""
import sys, json, time, random, os
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer

data_path, obj, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
B, LR, W, L, DEPTH, MAXLEN, EVAL = int(os.environ.get("B", 256)), float(os.environ.get("LR", 5e-4)), 256, 4, 4, 32, 500
dev = "mps"; torch.manual_seed(seed); random.seed(seed)
d = torch.load(data_path); feats, caps = d["feats"], d["caps"]
n = len(caps); ntr, nva = int(n * .8), int(n * .9)
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
PAD, MASK, CLS, SEP = tok.pad_token_id, tok.mask_token_id, tok.cls_token_id, tok.sep_token_id
def ids(texts): return [tok(t, add_special_tokens=False, truncation=True, max_length=MAXLEN - 2)["input_ids"] for t in texts]
cap_ids = [ids(c) for c in caps]
def batchify(seqs, mask_p=0.0, span=False):
    out = []
    for s in seqs:
        if span and len(s) > 3:
            a = random.randint(0, len(s) - 3); b = random.randint(a + 3, len(s)); s = s[a:b]
        s = [MASK if random.random() < mask_p else t for t in s]
        out.append([CLS] + s + [SEP])
    m = max(map(len, out)); return torch.tensor([x + [PAD] * (m - len(x)) for x in out], device=dev)

class Enc(nn.Module):
    def __init__(s, causal):
        super().__init__(); s.causal = causal
        s.emb = nn.Embedding(tok.vocab_size, W); s.pos = nn.Embedding(MAXLEN, W)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(W, 4, W * 4, 0.0, batch_first=True, norm_first=True), DEPTH)
        s.ln = nn.LayerNorm(W)
    def hidden(s, x):
        T = x.shape[1]; h = s.emb(x) + s.pos(torch.arange(T, device=x.device))
        cm = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1) if s.causal else None
        return s.ln(s.tf(h, mask=cm, src_key_padding_mask=(x == PAD)))
    def forward(s, x):
        h = s.hidden(x); m = (x != PAD).unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1)
enc = Enc(obj == "ntp").to(dev)
proj = nn.Sequential(nn.Linear(W, 512), nn.GELU(), nn.Linear(512, 512)).to(dev)
params = list(enc.parameters()) + list(proj.parameters())
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)

def vicreg(a, b):  # same anti-collapse for good and bad: only the positives differ
    inv = F.mse_loss(a, b)
    def reg(z):
        z = z - z.mean(0); std = (z.var(0) + 1e-4).sqrt(); var = F.relu(1 - std).mean()
        c = (z.T @ z) / (len(z) - 1); off = c - torch.diag(torch.diag(c))
        return var, (off ** 2).sum() / z.shape[1]
    va, ca = reg(a); vb, cb = reg(b)
    return 25 * inv + 25 * (va + vb) + 1.0 * (ca + cb)

# ---- eval ----
def embed_text(idx):
    enc.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 256): out.append(enc(batchify([cap_ids[j][0] for j in idx[i:i+256]])).cpu())
    enc.train(); return torch.cat(out)
def mknn(a, b, k=10):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1); N = len(a)
    def nn_(x): s = x @ x.T; s.fill_diagonal_(-9); return s.topk(k, 1).indices
    na, nb = nn_(a), nn_(b)
    return float(sum(len(set(na[i].tolist()) & set(nb[i].tolist())) for i in range(N)) / (N * k))
def procrustes(a, b):  # orthonormal-rows map taking a (W) to b (D), after centering + whitening scale
    am, bm = a.mean(0), b.mean(0); u, _, vt = torch.linalg.svd((a - am).T @ (b - bm), full_matrices=False)
    return am, bm, u @ vt
def evaluate():
    va, te = list(range(ntr, nva)), list(range(nva, n))
    tv, tt = embed_text(va), embed_text(te); iv, it = feats[va], feats[te]
    am, bm, R = procrustes(tv, iv)
    s = F.normalize((it - bm), dim=1) @ F.normalize((tt - am) @ R, dim=1).T  # image -> caption
    r1 = float((s.argmax(1) == torch.arange(len(te))).float().mean())
    return {"mknn": mknn(tt, it), "r1": r1}

log, t0 = [], time.time(); print(obj, seed, "start", flush=True)
for step in range(1, STEPS + 1):
    idx = random.sample(range(ntr), B)
    if obj == "good":
        pairs = [random.sample(cap_ids[i], 2) for i in idx]
        x1, x2 = batchify([p[0] for p in pairs]), batchify([p[1] for p in pairs])
        loss = vicreg(proj(enc(x1)), proj(enc(x2)))
    elif obj == "bad":
        c = [random.choice(cap_ids[i]) for i in idx]
        x1, x2 = batchify(c, 0.15, True), batchify(c, 0.15, True)
        loss = vicreg(proj(enc(x1)), proj(enc(x2)))
    else:
        x = batchify([random.choice(cap_ids[i]) for i in idx]); h = enc.hidden(x)[:, :-1]
        loss = F.cross_entropy((h @ enc.emb.weight.T).reshape(-1, tok.vocab_size), x[:, 1:].reshape(-1), ignore_index=PAD)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    if step % EVAL == 0 or step == STEPS:
        r = evaluate(); r.update(step=step, loss=float(loss), sec=round(time.time() - t0)); log.append(r); print(r, flush=True)
OUT = os.environ.get("OUT", "results"); os.makedirs(OUT, exist_ok=True)
json.dump({"obj": obj, "seed": seed, "steps": STEPS, "lr": LR, "n": n, "log": log}, open(f"{OUT}/{obj}_s{seed}_lr{LR:g}.json" if OUT != "results" else f"{OUT}/{obj}_s{seed}.json", "w"))
