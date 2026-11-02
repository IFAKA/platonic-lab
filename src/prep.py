"""Fetch N COCO images + their 5 human captions, embed images once with frozen DINOv2.
Usage: prep.py N   -> data/coco_{N}.pt  (feats [N,D], caps list[list[str]])"""
import sys, io, json, urllib.request, concurrent.futures as cf, time
import torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

N = int(sys.argv[1]); t0 = time.time()
API = "https://datasets-server.huggingface.co/rows?dataset=yerevann/coco-karpathy&config=default&split={s}&offset={o}&length=100"
def get(u, tries=4):
    for i in range(tries):
        try: return urllib.request.urlopen(u, timeout=30).read()
        except Exception: time.sleep(1 + i)
    return None
rows = []
with cf.ThreadPoolExecutor(8) as ex:
    pages = list(ex.map(lambda o: get(API.format(s="train", o=o)), range(0, N, 100)))
for p in pages:
    if p: rows += [r["row"] for r in json.loads(p)["rows"]]
rows = rows[:N]
print("rows", len(rows), f"{time.time()-t0:.0f}s", flush=True)
with cf.ThreadPoolExecutor(16) as ex:
    blobs = list(ex.map(lambda r: get(r["url"]), rows))
ok = [(r, b) for r, b in zip(rows, blobs) if b]
print("images", len(ok), f"{time.time()-t0:.0f}s", flush=True)
dev = "mps"
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
m = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
feats = []
with torch.no_grad():
    for i in range(0, len(ok), 64):
        ims = [Image.open(io.BytesIO(b)).convert("RGB") for _, b in ok[i:i+64]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        feats.append(m(pixel_values=x).pooler_output.float().cpu())
        if i % 1280 == 0: print("embed", i, f"{time.time()-t0:.0f}s", flush=True)
caps = [[s.strip() for s in r["sentences"][:5]] for r, _ in ok]
keep = [i for i, c in enumerate(caps) if len(c) == 5]
torch.save({"feats": torch.cat(feats)[keep], "caps": [caps[i] for i in keep]}, f"data/coco_{N}.pt")
print("saved", len(keep), f"{time.time()-t0:.0f}s")
