# Neural Debris NDR Frontier V8

Private Kaggle GPU experiment bundle built after the exact NDR229 reproduction.

The notebook trains four classification-head repairs using only the supplied
poisoned model, the public unlearn set, and deterministic synthetic streak
controls. It freezes model and post-processing selection before reading test
images, then exports:

- `submission_best.csv`
- `submission_diverse.csv`
- `submission_ndr229_control.csv`
- `submission.csv` as an alias of the locally selected best finalist

It never calls the Kaggle submission API.

The selection objective is lower-is-better maCADD on public-only pseudo-clean
and synthetic-retention references. Leaderboard scores and test predictions are
not used to choose a model or post-processing configuration.
