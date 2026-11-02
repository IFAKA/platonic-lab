# platonic-lab brief
What: a laptop test of a viral claim ("how you pair training data beats model size") from a LeCun/Balestriero paper.
For: devs who follow AI research and like checks of hype.
The wow: with human captions only, 3 small text models, same data and time: two-different-captions 0.32 vs
  hidden-words 0.19 vs next-word prediction 0.15 (match with DINOv2, chance 0.01). 3 seeds, same ordering every seed.
Proof: results/chart.png, results/table.md; code in this folder (no public repo URL yet).
Honest gaps: ~9M-param models, 8k photos, one dataset; paper's big run used AI-written captions; R@1 gap smaller;
  tuned table is 1 seed; not tested at 72B.
Pillar: "how I make agents do it" / small checks (agent built and ran it end to end in ~3 hours).
