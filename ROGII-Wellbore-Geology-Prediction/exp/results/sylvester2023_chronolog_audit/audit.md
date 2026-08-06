# Sylvester (2023) ChronoLog method audit

Primary paper: Z. Sylvester, *Automated multi-well stratigraphic correlation
and model building using relative geologic time*, Basin Research 35,
1961--1984, DOI 10.1111/bre.12787. The author's paper page links only the
`chronolog_data` repository (681 gamma-ray logs); the ChronoLog Python module
itself is not in that repository and was not found among the author's public
GitHub repositories.

## Exact reproducible details

### Per-log normalization

For each log independently, choose lower and upper percentiles `q_lo` and
`q_hi`, subtract the lower quantile, divide by their difference, and clip:

```python
lo, hi = np.nanpercentile(x, [1, 99])
xn = np.clip((x - lo) / max(hi - lo, eps), 0.0, 1.0)
```

The final article describes 1st and 99th percentiles as the example/default
choice. The preprint only says lower and upper percentiles, so 1/99 should be
treated as the paper's concrete implementation setting rather than a theorem.

### Pairwise robust DTW

Construct the full local-cost matrix directly, rather than asking librosa for
Euclidean costs:

```python
C = np.abs(xn[:, None] - yn[None, :]) ** 0.15
D, wp = librosa.sequence.dtw(C=C, subseq=False, backtrack=True)
wp = wp[::-1]
```

Equation (1) is `d = abs(l2-l1)**alpha`; the stated good default and study
value is `alpha=0.15`. This is an outlier-resistant power cost, not Huber or
Student-t loss.

The paper assumes the first and last samples of every pair correlate. It does
not specify a Sakoe-Chiba band, Itakura slope, custom step penalties, or a
maximum warp radius. Therefore faithful librosa defaults are:

- `subseq=False`, hence fixed endpoint-to-endpoint alignment;
- steps `(1,1)`, `(0,1)`, `(1,0)`;
- additive weights all zero and multiplicative weights all one;
- `global_constraints=False` (so `band_rad=0.25` is inactive).

Calling this “constrained DTW” should refer only to monotone fixed-endpoint
DTW. Adding a warp band or slope limit is a new method, not a recovered
Sylvester parameter.

### Well graph

Make wells graph nodes and pairwise correlations edge attributes. The study
correlated edges within 1500 m, plus Delaunay edges to maintain longer-range
connectivity, producing 9,569 pairs. This spatial threshold is dataset-specific.

### Global loop-closing least squares

For every DTW-correlated pair `(well_a, i) <-> (well_b, j)`, depth shifts obey

`s[a,i] - s[b,j] = z[b,j] - z[a,i]`.

Stack these equations as `D_s s = dz`, where every row of sparse `D_s` has
one `+1` and one `-1`, then solve

`(D_s.T @ D_s) s = D_s.T @ dz`

with conjugate gradients, using a gauge constraint (for example fix one shift
to zero or add a zero-mean constraint) because only relative shifts are
identified. The paper avoids materializing the large design matrix and solves
the sparse normal system. It also reports a speed approximation: resample all
logs to the same relatively small length and assign pairwise depth differences
to the same starting index in both wells even when the precise partner index
differs; interpolate solved shifts back to original resolution.

Then `RGT(z)=z+s`. Enforce nondecreasing RGT by a cumulative maximum. Flat
segments created this way represent locally compressed intervals. Resample all
RGT logs to a common grid and average across wells to obtain the type log.

### CWT hierarchy

Apply a continuous wavelet transform with Ricker wavelets to the mean/type log
in RGT space. Stack responses from high to low frequency. At a chosen width,
find zero crossings and trace each crossing toward the highest-frequency
(width-1) side; this makes coarse boundaries subsets of finer boundaries.

Concrete study values:

- Ricker width 1 sample: 2,585 units;
- Ricker width 4 samples: 1,223 units;
- width 4 was used for the final gridded model;
- the figure also illustrates width 8, but the paper does not promote it as
  the selected modeling scale.

Thus “scale 4” is four samples on the common RGT/type-log grid, not four feet
or a universal geological wavelength.

## Competition disposition

The exact useful novelty is global loop reconciliation, not another pairwise
DTW cost. It requires a connected graph of multiple logs representing the same
stratigraphic interval. A legal competition adaptation would have to build the
graph only from training/typewell information inside each whole-well fold and
then map the held-out horizontal suffix without using its TVT. Existing local
pairwise DTW/path matching failures do not automatically test this global
loop-closing formulation, but the mapping from the reconciled vertical-well
RGT grid to a held-out horizontal trajectory remains the key unresolved step.
No submission was made.
