# Current public/research scan — 2026-08-04

Fresh search of current public Kaggle notebooks/discussions and primary LWD
gamma-ray literature. No submission was made.

## New primary paper

Long et al., *Automatic Optimization Method of Horizontal Well Formation Model
Based On Natural Gamma While Drilling*, Applied Radiation and Isotopes 236
(2026) 112743, DOI 10.1016/j.apradiso.2026.112743.

- https://www.sciencedirect.com/science/article/pii/S0969804326003271
- https://ssrn.com/abstract=5722645

The paper constructs a local geological model over each tool's detection
footprint, calculates natural-GR response with MCNP or azimuthal sensitivity
matrices, generates folded constant-thickness layer models, and optimizes
formation displacement curves by matching simulated and measured horizontal GR
as an unconstrained inverse problem.

This supports our physical forward-operator/candidate-path branch, but is not a
new independent lever: the ledger already contains nonstationary response
integration, candidate likelihoods, PF/beam path optimization, and smooth
structural priors. Exact objective/optimizer details were not accessible, so no
unspecified implementation was guessed and no duplicate pilot was started.

## Public notebook audit

`daniilkrasnovvv/rogii-solution-on-6-390-in-lb` is not evidence of a clean new
6.390 method. Its text says the visible parent is 6.568 and the variant is
unscored. It mounts three external artifact/model datasets, applies a
leaderboard-derived constant shift on one target well, and changes two
provenance/branch failures from `raise` to `print("Wertiba")`.

https://www.kaggle.com/code/daniilkrasnovvv/rogii-solution-on-6-390-in-lb

`yusuketogashi/rogii-another-approach` reports a best 6.858. Its isolatable
mechanism is the direction-free bimodal PF midpoint hedge, already recorded in
the ledger. The full notebook mounts eight external datasets/model packages
and continues with single-well leaderboard probes.

https://www.kaggle.com/code/yusuketogashi/rogii-another-approach

## Decision

No genuinely new legal branch survived the novelty/provenance gate. The 2026
paper validates the physical inverse-model direction already in flight; the
apparent public sub-7 discoveries duplicate the existing hedge or rely on
external artifacts and leaderboard-local shifts. Compute was not spent on a
redundant pilot.
