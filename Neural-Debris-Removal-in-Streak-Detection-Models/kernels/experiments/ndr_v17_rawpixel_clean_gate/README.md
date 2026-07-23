# NDR V17 - raw-pixel clean/poison gate

Independent GroupNorm CNN trained on identically composited raw grayscale
patches from public poison signals, public StreaksYolo clean streaks, and an
analytic Gaussian-PSF simulator. Two held-out transfer directions must pass
before any exact V12/M1 confidence can be reduced.

The notebook never adds or moves boxes, never raises confidence, creates no test
pseudo-labels, and never creates a competition submission.
