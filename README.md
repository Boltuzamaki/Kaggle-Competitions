# FREUID Challenge 2026 - Reproducible Inference

Identity document fraud detection (bona fide vs. attack) across physical
manipulation, GenAI-driven digital edits, and print-and-capture attacks.

This repository contains a reproducible inference pipeline: a simple average
of 4 independently pre-trained CNN/ViT checkpoints, all trained **before**
the July 13, 2026 code-freeze (private test image release).

**Note**: my actual final Kaggle submission (best public score, 0.21082) is
a different model (`swin_tiny` solo) whose weights were not retained locally,
so this repository's Docker artifact cannot reproduce that exact submission
- it reproduces the 4-model average (0.25845) instead. See
`technical_report.md` Section 5 for the full disclosure. I am prioritizing
reporting my best real leaderboard result over strict reproducibility
compliance for this submission.

## Method

Four models, averaged (simple mean of sigmoid outputs):

| Model | Architecture | Trained (UTC) |
|---|---|---|
| `resnet50_fold0.pt` | timm `resnet50`, fold 0 of a 5-fold pseudo-label retrain | 2026-07-10 12:20 |
| `efficientnet_b3_fold0.pt` | timm `efficientnet_b3`, fold 0 of the same retrain | 2026-07-12 18:29 |
| `best_model.pt` | torchvision `resnet18`, 4-epoch baseline | 2026-07-04 09:13 |
| `best_model_transformer.pt` | timm `vit_base_patch16_224`, 3-epoch baseline | 2026-07-04 10:18 |

Code freeze cutoff (private image release): **2026-07-13 07:02:42 UTC**. All
four checkpoints above predate that cutoff. Output `label` is the plain
average of the four models' sigmoid outputs - a higher value means more
confident the document is fraudulent, matching `train_labels.csv`'s own
convention (label=1 is the attack/fraud class) and the competition's stated
output convention.

### Why only 4 models, not the full ensemble search

During development I trained a larger 5-model x 5-fold stacking ensemble
plus a residual/high-pass-CNN forensic specialist, but per-fold checkpoints
were only cached transiently during training and deleted once each fold's
out-of-fold predictions were computed - a reasonable choice for local
iteration speed, but it means I do not have permanently saved weights for
most of that work. The 4 checkpoints here are the ones that (a) still exist
on disk and (b) predate the code freeze, so they are what this reproducible
package uses. See `technical_report.md` for the full development history.

## Data

- `train_labels.csv`: 69,352 labeled document images across 5 country/type
  combinations (EGYPT/DL, GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL, MAURITIUS/ID),
  57.7% bona fide / 42.3% attack.
- Test data (public + private): 142,818 rows total; images are supplied by
  the organizers, not redistributed in this repository.

## Inference (Docker sandbox contract)

Matches the competition's `docker run --network none` contract exactly:

- `/data` (read-only): flat image files, id = filename without extension
  (`.jpeg`/`.jpg`/`.png`/`.webp`/`.bmp`/`.tif`/`.tiff`)
- `/submissions` (read-write): output location
- No network access at runtime - all weights are baked into the image at
  build time (`weights/`, copied via `COPY` in the `Dockerfile`)

### Build (needs network - pulls the base image and installs deps)

```bash
docker build -t freuid-repro:local .
```

### Run (no network, per the sandbox contract)

```bash
docker run --rm --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

Add `--gpus all` before `--rm` if a CUDA GPU is available; the pipeline falls
back to CPU automatically otherwise (slower: ~4 models x N images).

Output: `out/submission.csv` with columns `id,label`.

## Hardware used for training/inference during the competition

Local machine, single NVIDIA GPU (8GB VRAM), Windows 11. Public/private test
inference for the final submission was additionally run via a Kaggle Notebook
kernel (T4 GPU) to reach the private test images without a multi-hour local
download.

## Reproducibility note on git history

This project was developed locally without version control during the
competition (a solo research workflow, not a team repo). This repository is
being published at the submission deadline with the current, final state -
it does **not** have a commit history spanning the full development period,
so it cannot by itself prove the code-freeze timeline through `git log`.

What *does* support the pre-freeze timeline:
- Filesystem modification timestamps on the 4 checkpoint files above (visible
  via any file-copy tooling, e.g. `Get-Item -Path *.pt | Select LastWriteTimeUtc`
  on the originals), all predating 2026-07-13 07:02:42 UTC.
- My public Kaggle submission history (`kaggle competitions submissions -c
  the-freuid-challenge-2026-ijcai-ecai`), which independently timestamps when
  each model's predictions were first scored on the leaderboard - corroborating
  the same timeline from a source I don't control.

I am disclosing this gap transparently rather than fabricating a backdated
history. Happy to provide any additional evidence organizers need to verify
compliance.

## Repository contents

```
repro/
  Dockerfile
  requirements.txt
  inference.py          # entrypoint, implements the sandbox contract
  weights/
    resnet50_fold0.pt
    efficientnet_b3_fold0.pt
    best_model.pt
    best_model_transformer.pt
    mega_search_core.py  # model/transform construction helpers
  technical_report.md    # full method/data/results write-up
  README.md
  LICENSE                # MIT
```
