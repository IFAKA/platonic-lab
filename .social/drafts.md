# Drafts (not posted). Attach results/chart.png to every post.

## X, angle 1: the fact check (recommended)
A viral post said LeCun's new paper proves model size doesn't matter, how you pair the training data does.

I checked on my laptop, with human-written captions only (the paper's big run used AI-written ones).

Match with an image model, 3 small text models, same data and time:
two different captions of one photo: 0.32
one caption, words hidden: 0.19
next-word prediction: 0.15

Same ordering in all 3 seeds. Limits in the thread.

## X, angle 2: the number
Same text model, same 8000 photo captions, same 3000 steps.
Only the training pairs change.

Next-word prediction: 0.15 match with DINOv2.
Two different captions of one photo: 0.32.

More than 2x, from the pairing alone. It held in 3 of 3 seeds. Chance is 0.01.

## X, angle 3: the limits (thread reply or later post)
What my check does not show:
- models are ~11M params, not 72B
- good pairs need to know which captions describe the same photo, and people wrote those looking at the photo
- retrieval (R@1) gap is smaller: 0.084 vs 0.057
- one dataset, one image model
Code and the rule I wrote before running: [repo link]

## Reddit r/MachineLearning ([R] or [P])
Title: Small-scale check of "good views beat next-token prediction" (Balestriero & LeCun 2026), human captions only
Body: the paper claim in one line; setup (COCO 10k, DINOv2-base, 4-layer width-256 encoders, LeJEPA-style anti-collapse
 loss replaced by VICReg-style, same for good and bad views); table; limits as in README; link; ask: "what would you
 test next, e.g. does the gap shrink with model size?"
Note: I used VICReg-style variance/covariance loss, not the paper's SIGReg. Say so in the post.
