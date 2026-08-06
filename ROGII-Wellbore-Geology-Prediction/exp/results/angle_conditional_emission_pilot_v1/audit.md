# Angle-conditioned probabilistic GR domain-transfer audit

## Primary-method basis

Winkler's *Geosteering using Bayesian Networks* explicitly separates true
formation gamma `M` from measured LWD gamma `L`. Its conditional table
`P(L|M)` represents tool eccentering, temperature, calibration and other
measurement effects. The demonstration uses a Gaussian centered at `M`, but
the discussion recommends replacing it with an empirical histogram of LWD
minus vertical-log differences collected over matched intervals. This is a
direct primary-source prescription for a learned vertical-to-LWD likelihood,
not a deterministic convolution or embedding-distance model.

The paper also represents `P(M|Y,Z)` as a distribution around the pilot-log
value at relative stratigraphic depth, and performs exact variable elimination
over trajectory and structure. Its synthetic settings are 10-ft lateral
spacing, 1-ft depth and 1-gAPI discretization, and initially 1-gAPI Gaussian
measurement noise; 5-gAPI noise is used in a sensitivity experiment. These are
illustrative, not universal tool parameters.

Alyaev and Elsheikh's peer-reviewed *Direct Multi-Modal Inversion of
Geophysical Logs Using Deep Learning* provides the richer alternative: a
mixture-density DNN trained with multiple-trajectory-prediction loss outputs
several stratigraphic trajectories and their probabilities, specifically for
real-time gamma-ray inversion. It addresses inverse multimodality, whereas the
present question is the forward emission. A conditional normalizing flow or
MDN for `P(LWD_GR | vertical patch, angle)` would be a faithful adaptation but
is more complex than needed for a falsification pilot.

## Strict pilot

Implemented `exp/angle_conditional_emission_pilot_v1.py` as a distributional
forest approximation:

- target is normalized horizontal/LWD GR;
- inputs are typewell GR, first/second derivative, local contrast, signed and
  absolute candidate `dTVT/dMD`, and candidate curvature;
- one ExtraTrees forest estimates conditional mean and a second estimates log
  residual variance;
- both exclude the entire validation fold of wells;
- the likelihood ranks only three already-fixed complete paths: immutable
  level-2 center, accepted PF, and replacement PF;
- no validation TVT enters emission fitting or path selection.

On the fixed seed-2408 240-well pilot (1,152,578 suffix rows):

| metric | RMSE |
|---|---:|
| immutable level-2 center on this subset | 8.91849 |
| emission-selected complete path | 9.11481 |

Only 98/240 wells improved. Fold deltas were unfavorable in four of five
folds; only fold 2 improved, 7.65811 to 7.64226. The selector overwhelmingly
preferred the weaker accepted PF path (157 wells), showing that better
cross-well GR emission likelihood still does not identify the trajectory with
lower TVT error.

Decision: rejected. Do not expand, soften weights post hoc, or submit.

## Sources

- Hugh Winkler, *Geosteering using Bayesian Networks* (Geophysics preprint):
  https://hughw.net/geosteering-bn/geosteering-by-bayesian-network.pdf
- Sergey Alyaev and Ahmed H. Elsheikh, *Direct Multi-Modal Inversion of
  Geophysical Logs Using Deep Learning*, Earth and Space Science 2022:
  https://arxiv.org/abs/2201.01871
