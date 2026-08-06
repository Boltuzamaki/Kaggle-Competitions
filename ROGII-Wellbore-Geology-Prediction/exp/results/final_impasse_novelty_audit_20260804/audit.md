# Final clean-method novelty audit — 2026-08-04

Scope: current public Kaggle notebooks/discussions and recent primary research.
Required exclusions: packaged predictions/checkpoints as evidence, overlap IDs,
train-only formation surfaces, hardcoded wells, and leaderboard-only calibration.

## Current Kaggle material

- [ROGII Another Approach](https://www.kaggle.com/code/yusuketogashi/rogii-another-approach)
  now advertises 6.858, but latest source is an A22 single-well leaderboard probe
  derived from an existing prediction SHA. It mounts multiple model/prediction
  packages, states that all three scored wells overlap train, uses train-only
  formation surfaces, and applies a hardcoded shift to `00e12e8b`. It is not a
  clean from-raw method or honest-validation disclosure.
- [ROGII Contact and U Restore](https://www.kaggle.com/code/yaroslavkholmirzayev/rogii-contact-and-u-restore)
  mounts the same artifact families, enables a guarded overlap override, uses
  formation planes/ANCC, and describes score triangulation from prior public
  submissions. Excluded.
- `rogii-geologia-v92-geographic-restoration` has a zero-byte public script in
  the current Kaggle pull, so it discloses no reproducible method.
- [Measure your noise floor before believing a lever](https://www.kaggle.com/code/georgymamarin/measure-your-noise-floor-before-believing-a-lever)
  contributes useful evidence that tiny leaderboard changes are nondeterministic
  (roughly 0.03 ft scale for the studied kernels), but introduces no prediction
  model. This reinforces the project's conservative promotion gate.
- The July working note's clean grouped-CV mechanisms (surface identity, spatial
  transfer, GR alignment, PF, selectors) were already represented and tested in
  the ledger. Its best stated standalone clean estimator is 8.43, not a missing
  below-7 route.

## Primary research

- [Alyaev & Elsheikh (2022), direct multimodal inversion](https://doi.org/10.1029/2021EA002186):
  synthetic GR/SVD heatmaps, multi-trajectory prediction loss, and probabilistic
  modes. Covered here by cost-volume CNNs, alignment U-Nets, synthetic masking,
  multimodal curve mixtures, PF/beam search, and conditional curve priors.
- [Real-Time Gamma Ray Based Structural Modeling (2026)](https://doi.org/10.2118/230756-MS):
  lane detection plus forward structural projection is conceptually relevant,
  but the public abstract supplies no reproducible algorithm or honest benchmark.
  Its mechanism is already covered by GR cost-volume models, SDF/alignment U-Nets,
  beam/PF trackers, and structural-surface extrapolation experiments.
- [Knowledge-guided look-ahead LWD inversion (2026)](https://www.sciencedirect.com/science/article/pii/S1995822626002499)
  and [multi-task U-Net LWD inversion](https://doi.org/10.1016/j.petsci.2025.12.023)
  use electromagnetic look-ahead measurements unavailable in this competition.
  Their knowledge-guided residual/U-Net pattern has already been piloted with
  available GR/trajectory inputs.
- [DISTINGUISH (2025)](https://arxiv.org/abs/2503.08509) uses GAN geological
  realizations, forward-tool surrogates, ensemble updating, and drilling-control
  optimization. It requires simulator/tool responses absent here; synthetic
  pretraining, ensemble updating, and sequential inference analogues are already
  in the ledger.

## Conclusion

No newly disclosed clean, reproducible, from-raw mechanism remains untested.
The only current public score below 7 located in this audit depends on expressly
excluded artifacts, overlap, train-only formations, hardcoding, and leaderboard
probing. No pilot was launched because every legal research mechanism either
requires unavailable modalities/simulators or maps to a completed experiment
family. No submission was made.
