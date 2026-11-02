"""Build results/table.md and results/chart.png from results/*.json (main) and results_tuned/*.json (tuned)."""
import json, glob, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
NAMES = {"good": "A: two different captions\n(good views)", "bad": "B: one caption, words\nhidden (bad views)", "ntp": "C: next-word\nprediction"}
COL = {"good": "#2a9d8f", "bad": "#e76f51", "ntp": "#6c757d"}
def load(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f)); out.setdefault(d["obj"], []).append(d)
    return out
main, tuned = load("results/*.json"), load("results_tuned/*.json")
fin = lambda d, k: d["log"][-1][k]
rows = ["| Method | Setting | Seeds | mutual 10-NN (mean, range) | image->caption R@1 |", "|---|---|---|---|---|"]
for o in ["good", "bad", "ntp"]:
    for label, src in [("same speed (5e-4)", main), ("own best speed", tuned)]:
        runs = src.get(o, [])
        if not runs: continue
        m = [fin(r, "mknn") for r in runs]; r1 = [fin(r, "r1") for r in runs]
        lr = runs[0]["lr"] if "lr" in runs[0] else 5e-4
        rows.append(f"| {o} | {label}, lr {lr:g} | {len(runs)} | {st.mean(m):.3f} ({min(m):.3f}-{max(m):.3f}) | {st.mean(r1):.3f} |")
open("results/table.md", "w").write("\n".join(rows) + "\n\nChance: mutual 10-NN 0.010, R@1 0.001. Test set: 1000 held-out photos.\n"); print("\n".join(rows))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, o in enumerate(["good", "bad", "ntp"]):
    for off, src, alpha in [(-0.18, main, .55), (0.18, tuned, 1.0)]:
        v = [fin(r, "mknn") for r in src.get(o, [])]
        if not v: continue
        ax[0].bar(i + off, st.mean(v), 0.34, color=COL[o], alpha=alpha)
        ax[0].scatter([i + off] * len(v), v, color="k", s=10, zorder=3)
ax[0].axhline(0.01, ls="--", c="gray", lw=1); ax[0].text(2.45, 0.015, "chance", ha="right", fontsize=8, color="gray")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([NAMES[o] for o in ["good", "bad", "ntp"]], fontsize=8)
ax[0].set_ylabel("match with DINOv2 (mutual 10-NN)"); ax[0].set_title("Final score. Light: same speed, 3 seeds. Dark: own best speed.", fontsize=9)
for o in ["good", "bad", "ntp"]:
    for r in main.get(o, []):
        ax[1].plot([l["step"] for l in r["log"]], [l["mknn"] for l in r["log"]], color=COL[o], alpha=.7, label=NAMES[o].replace("\n", " ") if r is main[o][0] else None)
ax[1].set_xlabel("training steps"); ax[1].set_title("Score during training (same speed, 3 seeds)", fontsize=9); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/chart.png", dpi=160)
