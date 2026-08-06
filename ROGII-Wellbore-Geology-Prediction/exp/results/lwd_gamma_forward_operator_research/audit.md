# LWD natural-gamma forward response across dipping beds

Date: 2026-08-04

## Source-grounded physics

Natural gamma is a volume-weighted nuclear measurement, not a point sample.
Conaway (1980) defines the system response function as the noise-free tool
response to an infinitesimally thin radioactive zone under fixed tool,
borehole and formation conditions. Gadeken et al.'s gamma-tool patent, based
on Czubek's SPWLA response theory, states that a practical natural-GR vertical
response is well approximated by a finite detector box convolved with a
two-sided cusp. It gives a passive-GR cusp attenuation parameter of
`alpha = 3.0 / ft`; its worked geometry uses a 12-inch detector, and another
example describes a 0.75-ft detector with approximately 1-ft spatial
resolution.

Use the normalized response

```text
cusp_a(x) = (a/2) exp(-a |x|)
box_L(x)  = 1/L for |x| <= L/2, else 0
k_sym     = normalize(cusp_a * box_L)
```

The convolution is approximately Gaussian-like. A useful Gaussian surrogate
matches its second moment:

```text
sigma_s^2 = 2/a^2 + L^2/12
```

For `a=3/ft, L=1 ft`, `sigma_s=0.553 ft` and Gaussian-equivalent FWHM is
about 1.30 ft. Published tools vary: ODP documentation gives 38 cm for its CDR
natural-GR tool and 305 mm for an MCG probe, while operational summaries often
quote roughly 18--36 inches. The competition tool is unidentified, so these
are priors/ranges, not known metadata.

Shoulder-bed response follows directly: near a boundary, the kernel overlaps
both beds and the reading is a weighted average; a bed thinner than the
response width cannot reach its intrinsic plateau. Modern LWD gamma
spectroscopy forward modeling independently reports the same layer averaging
for thin and dipping beds and arbitrary trajectories.

## Dip geometry

Let `n` be the bedding normal, `u` the unit borehole tangent, and `H` true bed
thickness normal to bedding. Along-hole apparent thickness is

```text
H_app = H / abs(dot(u, n)).
```

Equivalently, if `beta` is the angle between borehole direction and the bed
plane, `H_app = H/abs(sin(beta))`. It diverges as the borehole becomes parallel
to bedding. In competition coordinates, candidate stratigraphic slope
`v=dTVT/dMD` plays the role of `dot(u,n)` up to TVT sign/scale, so
`H_app ~= H/abs(v)` and a fixed along-hole tool width maps to a local TVT width
`sigma_T = abs(v) sigma_s`.

## Compact nonstationary forward operator

The least-assumptive implementation works in measured-depth space and lets the
candidate TVT path create the nonstationarity:

```python
def forward_gr(md, tvt_path, tw_tvt, tw_gr, tau, kernel):
    # tau/kernel describe the along-hole detector response; sum(kernel)=1
    out = np.empty(len(md))
    for i, s in enumerate(md):
        tq = np.interp(s + tau, md, tvt_path,
                       left=tvt_path[0], right=tvt_path[-1])
        intrinsic_on_support = np.interp(tq, tw_tvt, tw_gr)
        out[i] = kernel @ intrinsic_on_support
    return out
```

This computes

```text
GR_hat(s_i) = integral k(tau) g_type(TVT(s_i+tau)) d tau.
```

It automatically produces slope-dependent stretching, shoulder mixing, and
curvature effects without manually varying a smoothing sigma. Suggested fixed
quadrature is `tau=-3..3 ft` at 0.1-ft spacing. A small, source-grounded grid is
`L in {0.5,0.75,1.0,1.5} ft` and `a in {2,3,4}/ft`, selected only on the visible
prefix or nested whole-well CV.

The typewell GR has already been blurred by its own tool. Strictly, let
`g_type = k_TW * g_intrinsic`; one should Wiener-deconvolve with a conservative
noise floor, then apply the LWD kernel. Without known tool metadata, direct use
of `g_type` in the integral is safer but can only add blur. A Gaussian shortcut
can add only the positive variance difference

```text
sigma_extra_T^2 = max(v^2 sigma_LWD_s^2 - sigma_TW_T^2, 0).
```

When this is negative the lateral response should be sharper than the
typewell; smoothing cannot represent it, and regularized deconvolution is
required.

## Asymmetry

For a centered total-GR detector in isotropic, plane-parallel beds, the
box-plus-cusp spatial kernel is symmetric. Dip by itself stretches the response
but does not justify a generic leading/trailing skew. Directional/azimuthal GR
can be asymmetric between up/down sectors, and Yuan et al. (2015) explicitly
use differing up/bottom response points to infer relative dip. Borehole
eccentering and 3-D attenuation can also create azimuthal sensitivity.

A non-azimuthal recorded log can acquire causal asymmetry from logging motion,
sampling and electronic time averaging. EPA guidance confirms logging speed
and time/sample constant affect resolution. If tested, keep this separate from
formation physics:

```text
k_recorded = k_sym * [exp(-tau/lambda)/lambda for tau>=0]
```

with the causal direction reversed when replaying an opposite logging
direction. `lambda = speed * time_constant`; because neither is provided, fit
only a very small prefix-locked grid (`0, 0.25, 0.5, 1 ft`). A free split-cusp
skew is less physically identified and should not be promoted without robust
prefix evidence.

## Public ROGII implementation audit

Searches of pulled public notebooks and indexed Kaggle code found PF/beam GR
misfit, Gaussian/rolling denoising, affine heel calibration, DTW and generic
multi-scale smoothing. None implements the exact detector-length box convolved
with a `3/ft` cusp, candidate-path quadrature `g(TVT(s+tau))`, optional
typewell-response deconvolution, or a separately modeled causal acquisition
kernel. Thus the exact forward operator is genuinely distinct from public
ROGII code. A local `nonstationary_lwd_operator_v1.py` experiment now exists in
the shared workspace, but it is local follow-up work, not evidence of prior
public implementation.

## Primary/authoritative sources

- Conaway (1980), *Direct determination of the gamma-ray logging system
  response function in field boreholes*, Geoexploration 18, 187--199,
  DOI 10.1016/0016-7142(80)90030-7.
- Conaway (1981), *Deconvolution of gamma-ray logs in the case of dipping
  radioactive zones*, Geophysics 46, 198--208, DOI 10.1190/1.1441189.
- Gadeken et al., US Patent 5,672,867, *Method for filtering gamma ray well
  logging tool response to enhance vertical detail while suppressing
  statistical noise* (box-plus-cusp response, `a=3/ft`, detector examples).
- Yuan et al. (2015), *A novel method for quantitative geosteering using
  azimuthal gamma-ray logging*, Applied Radiation and Isotopes 96, 16--23,
  PMID 25479436.
- Mendoza et al. (2015), *Fast numerical simulation of logging-while-drilling
  gamma-ray spectroscopy measurements*, Geophysics,
  DOI 10.1190/GEO2014-0584.1.
- ODP/IODP tool documentation for natural-GR vertical-resolution examples.

No submission was made.
