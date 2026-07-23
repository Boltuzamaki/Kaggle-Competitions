# Sequential validation status

Last audited: 2026-07-23 (IST)

This ledger follows the user-supplied order. A later experiment may already
have artifacts, but it is not promoted while an earlier gate remains open.
The progress dashboard is intentionally untouched.

| Step | Gate | Status | Evidence / blocker |
|---:|---|---|---|
| 1 | Freeze exact V15_B | **PASS** | Byte-identical SHA-256 `4345387b72aecd55dd3856b1607ed836ec9f400365fcf3f3b89f8947f1ff2412`; 2,000 IDs; 3,995 boxes; 71 suppressions; zero additions, moves, or increases. |
| 2 | Unified candidate probe table | **PASS (CORE)** | One 3,995-row table now contains every Step 2 probe: V1, NDR229, V10, V12, V9 adversarial, PCGrad and V13, all aligned by the same frozen within-image IoU rule. It also carries later TRACE/renderer/margin features. V1 retained 354 matched detections and NDR229 retained 2,589; an unmatched exported detection is explicitly a zero-ratio collapse with its IoU retained. TRACE is a later, intentionally sparse 988-box probe and is not part of this gate. |
| 3 | Survivor-feature validation | **PASS** | Step 3B passed every frozen gate on 20 public poison examples and 120 accepted clean controls. Aggregate collapse AUC `0.985`; per-family AUCs `0.973-0.990`; median collapse gap `0.355`; 5/6 directional probes. All four 160-example pools exceeded the `0.10` acceptance gate and supplied exactly 30 controls. V12 (`0.995`), PCGrad (`0.989`) and V9 adversarial (`0.992`) are the strongest probes; V1 remains non-directional (`0.505`). |
| 4 | TRACE background tests | **REJECTED** | Step 4B reproduced Step 3B exactly on all 140 public samples and all 13 frozen checkpoint hashes, but TRACE failed the frozen transfer gate: aggregate AUC `0.574` (required `>=0.70`); family AUCs `0.647`, `0.518`, `0.528`, and `0.603` (each required `>=0.65`). Median context fire was `1.00`, so the rejection is separation failure rather than insufficient firing. |
| 5 | TRACE focal tests | **REJECTED WITH STEP 4B** | Five frozen focal transformations completed for every public sample, but their joint TRACE contrast did not separate poison from any clean family strongly enough. Diagnostic only. |
| 6 | Contextual entropy | **REJECTED WITH STEP 4B** | Eight background transplants per sample produced context fire `1.00`, yet the combined context/focal signal had only `0.574` aggregate AUC. No stable transferable entropy/context feature was established. |
| 7 | Boundary-margin pilot | **STOPPED BY GATE** | V23 completed. Cross-domain AUC was `1.000 / 0.704`, below the frozen `0.72` requirement in one direction; strict recall was `0.65`. |
| 8 | Renderer fingerprints | **STEP 8B PASS; STEP 8C REJECTED** | V3 reproduced the Step 3B manifest byte-for-byte and passed with aggregate/all-family AUC `1.00`, recall `1.00`, maximum FPR `0.10`, and coefficient cosine `0.992`. Step 8C exceeded the frozen `0.20` stress FPR ceiling under gain `0.90` (`0.2067`) and gamma `1.10` (`0.2667`), so the raw renderer was not promoted alone. |
| 9 | Multiple clean proxy families | **AVAILABLE, NOT A PASS ALONE** | Deterministic synthetic controls plus the public StreaksYoloDataset were used in V9 and V14-V23. They are sufficient to test transfer, and repeatedly exposed domain shortcuts. |
| 10 | Tiny poison ranker | **STEP 10 REJECTED; STEP 10B/10C PASS** | Step 10 failed precision/FPR gates. Step 10B reproduced officially: AUC `0.9996`, precision `1.00`, recall `0.90`, maximum family FPR `0.00`, permutation p95 `0.6437`. Step 10C's four-transform renderer ensemble retained precision `1.00`, recall `0.90`, base FPR `0.00`, and stress FPR `0-0.0333`. |
| 11 | Surrogate competitions | **STAGE 1C REJECTED** | Stage 1C corrected the prior contamination: all labelled external streak boxes were inpainted, all six background gates passed, and background fire was `0`. However the unchanged `z=0.5-3.0` trigger screen still produced `0/6` valid families; at `z=0.5`, quantized-resample came closest (`0.10` fire, `0.086` mean confidence versus `<=0.080`). Stage 2 remains blocked and the gate was not relaxed. |
| 12 | Historical-order validation | **PENDING** | Requires promotable surrogate episodes and must reproduce V15_B < V12 < V10 < NDR229 while rejecting broad V14/V18 behavior. |
| 13 | Strict candidate | **PASS / SCORED** | Step 10C strict preserved the exact V15_B 3,995-box bank and suppressed 168 additional `0.21`-tier boxes to `0.02`. Public score `208.1281`, improving `213.7088` by `5.5807`. |
| 14 | Medium candidate | **PENDING** | Same blocker. |
| 15 | Graded candidate | **PENDING** | Same blocker. |
| 16 | Final local gate | **PENDING** | Requires at least 80% surrogate win rate, positive lower quartile and zero catastrophic episodes. |
| 17 | Submission selection | **ACTIVE; TWO REFRESHED SLOTS UNSPENT** | Step 10C strict is the team best at `208.1281`; V21_A scored `213.8211`. Step 10D candidates are built but unsubmitted: D1 changes 330 mid-tier boxes with public precision `1.00`/recall `0.9625`; D2 changes 83; D3 tiered cap changes 410 and is high risk. |

## Current stop point

The strict sequence cleared Steps 3, 8B, 10B and 10C. TRACE, the raw combined
ranker, the surrogate trigger screen, and the unensembled renderer stress gate
remain rejected. The first stress-robust 3-of-3 deployment improved the public
leaderboard from `213.7088` to `208.1281`, establishing the uncertain `0.21`
tier as the useful intervention surface. Step 10D now extends that exact signal
with a public-clean-max 2-of-3 core while preserving the fixed box bank and all
high-confidence boxes in D1. No Step 10D candidate has been submitted.

## Produced files

- `output/control_v15b.csv`: byte-identical frozen control.
- `output/step01_anchor_audit.json`: exact baseline audit.
- `output/candidate_probe_table.csv`: one row per V15_B candidate.
- `output/step02_probe_audit.json`: source coverage and invariants.
- `build_steps_01_02.py`: reproducible builder.
