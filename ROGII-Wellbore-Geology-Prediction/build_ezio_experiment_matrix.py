"""Build five CPU and one GPU ROGII Kaggle experiments for EzioAuditore."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import build_stack_contact_experiment as stack_base


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "kernels" / "ezio_matrix"
OWNER = "boltuzmaki"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence, found {count}: {old[:100]!r}")
    return text.replace(old, new, 1)


def insert_before(text: str, marker: str, addition: str) -> str:
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError(f"Marker not found: {marker!r}")
    return text[:pos] + addition.rstrip() + "\n\n\n" + text[pos:]


def splice(text: str, start: str, end: str, replacement: str) -> str:
    lo = text.find(start)
    hi = text.find(end, lo + len(start))
    if lo < 0 or hi < 0:
        raise RuntimeError(f"Could not splice {start!r} -> {end!r}")
    return text[:lo] + replacement.rstrip() + "\n\n\n" + text[hi:]


def package(slug: str, title: str, intro: str, control: str, driver: str, gpu: bool) -> None:
    notebook = {
        "cells": [
            stack_base.base.markdown(intro),
            stack_base.base.code(control, hidden=False),
            stack_base.base.code(stack_base.base.PF),
            stack_base.base.code(stack_base.base.BEAM),
            stack_base.base.code(stack_base.base.SIGNALS),
            stack_base.base.code(driver),
        ],
        "metadata": stack_base.base.NOTEBOOK["metadata"],
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"ezio-{slug[-12:]}-{i}"
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{slug}:cell-{i}")

    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["rogii-wellbore-geology-prediction"],
        "kernel_sources": [],
    }
    out = OUT_ROOT / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / metadata["code_file"]).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


COMMON_INTRO = """Every run is self-contained and uses only the mounted competition data.
The honest fallback is the previously validated residual LightGBM stack. Each
experiment changes one correction layer and writes standardized validation,
contact, prefix, feature-importance, and submission-integrity reports."""


# CPU 1: exact control on the second account.
package(
    "rogii-e01-stack-contact-control",
    "ROGII E01 Stack Contact Control",
    "# ROGII E01: Residual Stack + Strict Contact Control\n\n" + COMMON_INTRO,
    stack_base.CONTROL,
    stack_base.DRIVER,
    gpu=False,
)


# CPU 2: choose the formation contact by visible-prefix RMSE, allow a graded
# 1-3 ft move, and boundary re-anchor only outside the train MD range.
control_e02 = replace_once(
    stack_base.CONTROL,
    'CONTACT_REFERENCE = "EGFDU"',
    'CONTACT_REFERENCE = "prefix_selected"\n'
    'CONTACT_REFERENCES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")\n'
    'CONTACT_GRADED_RMSE_LIMIT = 3.0\n'
    'CONTACT_GRADED_WEIGHT = 0.85\n'
    'CONTACT_REANCHOR_DECAY_MD = 800.0',
)

multi_contact_code = r'''
def contact_reconstruction(hw_train, tw_train, reference):
    if reference not in hw_train.columns:
        return None
    geology = tw_train.dropna(subset=["Geology", "TVT"]).copy()
    ref = geology.loc[geology["Geology"].astype(str) == reference, "TVT"]
    if ref.empty:
        return None
    ref_tvt = float(ref.min())
    raw = ref_tvt - (
        hw_train["Z"].to_numpy(float) - hw_train[reference].to_numpy(float)
    )
    valid = np.isfinite(raw) & np.isfinite(hw_train["TVT"].to_numpy(float))
    if valid.sum() < CONTACT_MIN_PHYS_ROWS:
        return None
    bias = float(np.mean(hw_train["TVT"].to_numpy(float)[valid] - raw[valid]))
    result = raw + bias
    return result if np.isfinite(result).sum() >= CONTACT_MIN_PHYS_ROWS else None


def apply_contact_guard(group):
    well = str(group["well"].iloc[0])
    train_hw_path = DATA / "train" / f"{well}__horizontal_well.csv"
    train_tw_path = DATA / "train" / f"{well}__typewell.csv"
    base = group["tvt_pre_contact"].to_numpy(float)
    report = {
        "well": well,
        "accepted": False,
        "reference": "",
        "weight": 0.0,
        "replaced_rows": 0,
        "reanchored_rows": 0,
    }
    if not train_hw_path.exists() or not train_tw_path.exists():
        report["reason"] = "no_same_well_train_copy"
        return base, report

    hw_test, _ = load_pair(well, "test")
    hw_train = pd.read_csv(train_hw_path)
    tw_train = pd.read_csv(train_tw_path)
    train_md = hw_train["MD"].to_numpy(float)
    known = hw_test[hw_test["TVT_input"].notna()].copy()
    in_range = known["MD"].between(np.nanmin(train_md), np.nanmax(train_md)).to_numpy()
    comparable = known.loc[in_range]
    if len(comparable) < CONTACT_MIN_PREFIX_ROWS:
        report["reason"] = "too_few_comparable_prefix_rows"
        report["prefix_rows"] = int(len(comparable))
        return base, report

    candidates = []
    truth = comparable["TVT_input"].to_numpy(float)
    known_md = comparable["MD"].to_numpy(float)
    for reference in CONTACT_REFERENCES:
        contact = contact_reconstruction(hw_train, tw_train, reference)
        if contact is None:
            continue
        finite = np.isfinite(contact)
        if finite.sum() < CONTACT_MIN_PHYS_ROWS:
            continue
        pred = np.interp(known_md, train_md[finite], contact[finite])
        score = pooled_rmse(truth, pred)
        candidates.append((score, reference, contact, finite))
    if not candidates:
        report["reason"] = "contact_unavailable"
        return base, report

    prefix_rmse, reference, contact, finite = min(candidates, key=lambda row: row[0])
    report.update(
        {
            "reference": reference,
            "prefix_rows": int(len(comparable)),
            "prefix_rmse": float(prefix_rmse),
            "candidate_references": int(len(candidates)),
        }
    )
    if prefix_rmse <= CONTACT_PREFIX_RMSE_LIMIT:
        weight = 1.0
        reason = "prefix_verified_full"
    elif prefix_rmse <= CONTACT_GRADED_RMSE_LIMIT:
        weight = CONTACT_GRADED_WEIGHT
        reason = "prefix_verified_graded"
    else:
        report["reason"] = "prefix_rmse_failed"
        return base, report

    hidden_idx = group["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
    hidden_md = hw_test["MD"].to_numpy(float)[hidden_idx]
    lo, hi = np.nanmin(train_md[finite]), np.nanmax(train_md[finite])
    inside = (hidden_md >= lo) & (hidden_md <= hi)
    lookup = np.interp(hidden_md, train_md[finite], contact[finite])
    output = base.copy()
    output[inside] = (1.0 - weight) * base[inside] + weight * lookup[inside]

    outside = ~inside
    if outside.any() and inside.any():
        edge = int(np.flatnonzero(inside)[-1])
        shift = float(output[edge] - base[edge])
        distance = np.abs(hidden_md[outside] - hidden_md[edge])
        decay = np.exp(-distance / CONTACT_REANCHOR_DECAY_MD)
        output[outside] = base[outside] + shift * decay

    report.update(
        {
            "accepted": True,
            "reason": reason,
            "weight": float(weight),
            "replaced_rows": int(inside.sum()),
            "reanchored_rows": int(outside.sum() if inside.any() else 0),
        }
    )
    return output, report
'''
driver_e02 = splice(
    stack_base.DRIVER,
    "def contact_reconstruction(hw_train, tw_train, reference):",
    "contact_rows = []",
    multi_contact_code,
)
package(
    "rogii-e02-multiref-contact",
    "ROGII E02 Multiref Contact",
    "# ROGII E02: Prefix-Selected Multi-Reference Contact\n\n" + COMMON_INTRO,
    control_e02,
    driver_e02,
    gpu=False,
)


# CPU 3: activate the existing conservative prefix-verified structural move.
control_e03 = replace_once(
    stack_base.CONTROL,
    'SUBMISSION_PROFILE = "contact_gated_anchor"  # previous_stack | contact_gated_anchor',
    'SUBMISSION_PROFILE = "visible_prefix_bounded"  # bounded prefix correction, then strict contact',
)
control_e03 = replace_once(
    control_e03,
    'if SUBMISSION_PROFILE not in {"previous_stack", "contact_gated_anchor"}:',
    'if SUBMISSION_PROFILE not in {"previous_stack", "contact_gated_anchor", "visible_prefix_bounded"}:',
)
package(
    "rogii-e03-prefix-bounded",
    "ROGII E03 Prefix Bounded",
    "# ROGII E03: Conservative Visible-Prefix Correction\n\n" + COMMON_INTRO,
    control_e03,
    stack_base.DRIVER,
    gpu=False,
)


# CPU 4: enable the already-audited heel datum bimodal hedge.
control_e04 = replace_once(
    stack_base.CONTROL,
    "APPLY_BIMODAL_HEDGE = False",
    "APPLY_BIMODAL_HEDGE = True",
)
package(
    "rogii-e04-heel-bimodal",
    "ROGII E04 Heel Bimodal",
    "# ROGII E04: Heel-Calibrated Bimodal Datum Hedge\n\n" + COMMON_INTRO,
    control_e04,
    stack_base.DRIVER,
    gpu=False,
)


# CPU 5: leave-one-well-out local formation plane, calibrated only on the
# visible prefix and blended conservatively into the honest fallback.
control_e05 = replace_once(
    stack_base.CONTROL,
    "PROJECTED_ANCHOR_WEIGHT = 0.60",
    "PROJECTED_ANCHOR_WEIGHT = 0.60\n"
    "SPATIAL_K = 12\n"
    "SPATIAL_WEIGHT = 0.15\n"
    "SPATIAL_PREFIX_RMSE_LIMIT = 8.0\n"
    "SPATIAL_DIFF_P95_LIMIT = 50.0",
)
spatial_code = r'''
# Leave-one-well-out local formation-surface candidate.
from scipy.spatial import cKDTree

SPATIAL_FORMS = ("EGFDU", "EGFDL", "BUDA")
spatial_rows = []
for _path in sorted((DATA / "train").glob("*__horizontal_well.csv")):
    _well = _path.name.split("__")[0]
    try:
        _cols = ["X", "Y"] + list(SPATIAL_FORMS)
        _frame = pd.read_csv(_path, usecols=_cols)
        _row = {
            "well": _well,
            "x": float(_frame["X"].median()),
            "y": float(_frame["Y"].median()),
        }
        for _form in SPATIAL_FORMS:
            _row[_form] = float(_frame[_form].median())
        spatial_rows.append(_row)
    except Exception:
        continue
spatial_index = pd.DataFrame(spatial_rows)
spatial_scale = spatial_index[["x", "y"]].std().replace(0, 1.0).to_numpy(float)
spatial_xy_scaled = spatial_index[["x", "y"]].to_numpy(float) / spatial_scale
spatial_tree = cKDTree(spatial_xy_scaled)


def spatial_plane_candidate(well, hw):
    qx, qy = float(hw["X"].median()), float(hw["Y"].median())
    query = np.array([qx, qy]) / spatial_scale
    k_query = min(len(spatial_index), SPATIAL_K + 4)
    distance, index = spatial_tree.query(query, k=k_query)
    distance = np.atleast_1d(distance)
    index = np.atleast_1d(index)
    keep = spatial_index.iloc[index]["well"].astype(str).to_numpy() != str(well)
    distance, index = distance[keep][:SPATIAL_K], index[keep][:SPATIAL_K]
    if len(index) < 5:
        return None, {"reason": "too_few_neighbors"}

    x_all = hw["X"].to_numpy(float)
    y_all = hw["Y"].to_numpy(float)
    z_all = hw["Z"].to_numpy(float)
    known_idx = np.flatnonzero(hw["TVT_input"].notna().to_numpy())
    if len(known_idx) < 80:
        return None, {"reason": "short_prefix"}
    cut = max(50, int(round(0.70 * len(known_idx))))
    fit_idx, hold_idx = known_idx[:cut], known_idx[cut:]
    if len(hold_idx) < 30:
        return None, {"reason": "short_holdout"}

    best = None
    for form in SPATIAL_FORMS:
        values = spatial_index.iloc[index][form].to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(distance)
        if valid.sum() < 5:
            continue
        neighbors = spatial_index.iloc[index[valid]]
        dx = neighbors["x"].to_numpy(float) - qx
        dy = neighbors["y"].to_numpy(float) - qy
        design = np.column_stack([dx, dy, np.ones(valid.sum())])
        weights = 1.0 / (distance[valid] + 0.05)
        coef = np.linalg.lstsq(design * weights[:, None], values[valid] * weights, rcond=None)[0]
        surface = (
            coef[0] * (x_all - qx) + coef[1] * (y_all - qy) + coef[2]
        )
        bias_fit = float(
            np.median(
                hw["TVT_input"].to_numpy(float)[fit_idx]
                + z_all[fit_idx]
                - surface[fit_idx]
            )
        )
        hold_pred = surface[hold_idx] + bias_fit - z_all[hold_idx]
        hold_truth = hw["TVT_input"].to_numpy(float)[hold_idx]
        score = pooled_rmse(hold_truth, hold_pred)
        candidate = (score, form, surface, float(np.min(distance)))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return None, {"reason": "no_valid_surface"}

    score, form, surface, neighbor_distance = best
    bias = float(
        np.median(
            hw["TVT_input"].to_numpy(float)[known_idx]
            + z_all[known_idx]
            - surface[known_idx]
        )
    )
    return surface + bias - z_all, {
        "reason": "candidate",
        "form": form,
        "prefix_rmse": float(score),
        "neighbor_distance": float(neighbor_distance),
        "neighbors": int(len(index)),
    }


spatial_reports = []
spatial_adjusted = []
for _, group in test.groupby("well", sort=False):
    group = group.copy()
    well = str(group["well"].iloc[0])
    hw, _ = load_pair(well, "test")
    candidate, report = spatial_plane_candidate(well, hw)
    base = group["tvt_honest"].to_numpy(float)
    report["well"] = well
    report["accepted"] = False
    if candidate is not None:
        hidden_idx = group["id"].str.rsplit("_", n=1).str[-1].astype(int).to_numpy()
        cand_hidden = candidate[hidden_idx]
        diff = cand_hidden - base
        diff_p95 = float(np.quantile(np.abs(diff), 0.95))
        report["diff_p95"] = diff_p95
        accepted = (
            report["prefix_rmse"] <= SPATIAL_PREFIX_RMSE_LIMIT
            and diff_p95 <= SPATIAL_DIFF_P95_LIMIT
        )
        if accepted:
            ramp = 1.0 - np.exp(-np.arange(len(base)) / max(80.0, 0.12 * len(base)))
            move = SPATIAL_WEIGHT * ramp * np.clip(diff, -25.0, 25.0)
            group["tvt_honest"] = base + move
            report["accepted"] = True
            report["max_abs_move"] = float(np.max(np.abs(move)))
    spatial_adjusted.append(group)
    spatial_reports.append(report)
test = pd.concat(spatial_adjusted, ignore_index=True)
pd.DataFrame(spatial_reports).to_csv(WORK / "spatial_surface_report.csv", index=False)
'''
driver_e05 = insert_before(
    stack_base.DRIVER,
    "def heel_calibration_and_scan(well, base_pred):",
    spatial_code,
)
package(
    "rogii-e05-spatial-surface",
    "ROGII E05 Spatial Surface",
    "# ROGII E05: Prefix-Gated Spatial Formation Surface\n\n" + COMMON_INTRO,
    control_e05,
    driver_e05,
    gpu=False,
)


# GPU 1: five-fold OOF residual stack plus a per-well candidate ranker. The
# ranker is deployed only if its OOF gain and worst-decile guard both pass.
control_e06 = replace_once(stack_base.CONTROL, "RUN_GROUP_CV = False", "RUN_GROUP_CV = True")
control_e06 = replace_once(control_e06, "LGB_ESTIMATORS = 1200", "LGB_ESTIMATORS = 800")
control_e06 = replace_once(
    control_e06,
    "STACK_CLIP = 60.0",
    "STACK_CLIP = 60.0\n"
    "GPU_RANKER = True\n"
    "RANKER_MIN_GAIN = 0.20\n"
    "RANKER_WORST_TOLERANCE = 0.05",
)
driver_e06 = replace_once(
    stack_base.DRIVER,
    "from sklearn.model_selection import GroupKFold, GroupShuffleSplit",
    "from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold",
)
driver_e06 = replace_once(
    driver_e06,
    "    random_state=2026,\n)",
    '    random_state=2026,\n'
    '    device_type="gpu" if GPU_RANKER and not LOCAL_DEBUG else "cpu",\n'
    ")",
)

ranker_code = r'''
# Per-well candidate ranker. Candidate labels and validation use only OOF predictions.
candidate_names = ["stack", "pf", "beam", "selector", "structural", "hold"]
candidate_train = {
    "stack": np.clip(oof * STACK_SHRINK, -STACK_CLIP, STACK_CLIP),
    "pf": train["pf_d"].to_numpy(float),
    "beam": train["beam_d"].to_numpy(float),
    "selector": (
        0.85 * train["pf_d"].to_numpy(float)
        + 0.15 * train["beam_d"].to_numpy(float)
    ),
    "structural": -train["d_z"].to_numpy(float),
    "hold": np.zeros(len(train), dtype=float),
}


def safe_mean(group, column):
    if column not in group:
        return 0.0
    values = group[column].to_numpy(float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.0


def make_well_features(frame, candidates):
    rows = []
    errors = {}
    row_index = {}
    truth = frame["target"].to_numpy(float) if "target" in frame else None
    for well, group in frame.groupby("well", sort=False):
        idx = group.index.to_numpy()
        row_index[str(well)] = idx
        pf = group["pf_d"].to_numpy(float)
        beam = group["beam_d"].to_numpy(float)
        row = {
            "well": str(well),
            "n_rows": float(len(group)),
            "md_span": float(group["d_md"].max() - group["d_md"].min()),
            "z_span": float(group["d_z"].max() - group["d_z"].min()),
            "gr_std": float(group["gr"].std()),
            "pf_beam_rmse": float(np.sqrt(np.mean((pf - beam) ** 2))),
            "pf_beam_bias": float(np.mean(pf - beam)),
            "pf_std": float(np.std(pf)),
            "beam_std": float(np.std(beam)),
            "ncc8": safe_mean(group, "ncc8_s"),
            "ncc15": safe_mean(group, "ncc15_s"),
            "ncc25": safe_mean(group, "ncc25_s"),
            "tortuosity": safe_mean(group, "tort"),
            "dz_abs_mean": float(np.mean(np.abs(group["dz_dmd"].to_numpy(float)))),
            "gr_texture": safe_mean(group, "gr_s21"),
        }
        if truth is not None:
            err = {}
            for name in candidate_names:
                pred = candidates[name][idx]
                err[name] = float(np.sqrt(np.mean((truth[idx] - pred) ** 2)))
                row[f"rmse_{name}"] = err[name]
            row["label"] = int(np.argmin([err[name] for name in candidate_names]))
            errors[str(well)] = err
        rows.append(row)
    return pd.DataFrame(rows), row_index, errors


ranker_wells, ranker_train_rows, candidate_errors = make_well_features(
    train, candidate_train
)
ranker_feature_cols = [
    c for c in ranker_wells.columns
    if c not in {"well", "label"} and not c.startswith("rmse_")
]
ranker_X = np.nan_to_num(
    ranker_wells[ranker_feature_cols].to_numpy(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
ranker_y = ranker_wells["label"].to_numpy(int)
ranker_oof_prob = np.zeros((len(ranker_wells), len(candidate_names)), dtype=float)
ranker_split = KFold(n_splits=5, shuffle=True, random_state=2026)
for fold, (fit_idx, valid_idx) in enumerate(ranker_split.split(ranker_X)):
    rank_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(candidate_names),
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        reg_lambda=5.0,
        reg_alpha=1.0,
        verbose=-1,
        random_state=2026 + fold,
    ).fit(ranker_X[fit_idx], ranker_y[fit_idx])
    fold_prob = rank_model.predict_proba(ranker_X[valid_idx])
    ranker_oof_prob[np.ix_(valid_idx, rank_model.classes_.astype(int))] = fold_prob

ranker_oof_delta = np.zeros(len(train), dtype=float)
for row_pos, row in ranker_wells.iterrows():
    idx = ranker_train_rows[str(row["well"])]
    for candidate_idx, name in enumerate(candidate_names):
        ranker_oof_delta[idx] += (
            ranker_oof_prob[row_pos, candidate_idx] * candidate_train[name][idx]
        )


def per_well_values(frame, truth, prediction):
    values = []
    for _, group in frame.groupby("well", sort=False):
        idx = group.index.to_numpy()
        values.append(float(np.sqrt(np.mean((truth[idx] - prediction[idx]) ** 2))))
    return np.asarray(values)


truth_delta = train["target"].to_numpy(float)
stack_delta = candidate_train["stack"]
stack_values = per_well_values(train, truth_delta, stack_delta)
ranker_values = per_well_values(train, truth_delta, ranker_oof_delta)
stack_pooled = pooled_rmse(truth_delta, stack_delta)
ranker_pooled = pooled_rmse(truth_delta, ranker_oof_delta)
stack_worst = float(np.quantile(stack_values, 0.90))
ranker_worst = float(np.quantile(ranker_values, 0.90))
ranker_gain = float(stack_pooled - ranker_pooled)
ranker_accepted = bool(
    ranker_gain >= RANKER_MIN_GAIN
    and ranker_worst <= stack_worst + RANKER_WORST_TOLERANCE
)

validation_rows = [
    {
        "candidate": "stack_oof",
        "pooled_rmse": stack_pooled,
        "mean_well_rmse": float(np.mean(stack_values)),
        "worst_decile_rmse": stack_worst,
        "accepted": True,
    },
    {
        "candidate": "ranker_oof",
        "pooled_rmse": ranker_pooled,
        "mean_well_rmse": float(np.mean(ranker_values)),
        "worst_decile_rmse": ranker_worst,
        "gain_vs_stack": ranker_gain,
        "accepted": ranker_accepted,
    },
]
for name in candidate_names:
    pred = candidate_train[name]
    vals = per_well_values(train, truth_delta, pred)
    validation_rows.append(
        {
            "candidate": name,
            "pooled_rmse": pooled_rmse(truth_delta, pred),
            "mean_well_rmse": float(np.mean(vals)),
            "worst_decile_rmse": float(np.quantile(vals, 0.90)),
            "accepted": name == "stack",
        }
    )
pd.DataFrame(validation_rows).to_csv(
    WORK / "candidate_ranker_validation.csv", index=False
)
ranker_wells.to_csv(WORK / "candidate_ranker_wells.csv", index=False)

ranker_full = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=len(candidate_names),
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    reg_lambda=5.0,
    reg_alpha=1.0,
    verbose=-1,
    random_state=2026,
).fit(ranker_X, ranker_y)

stack_test_d = learned_test_d.copy()
candidate_test = {
    "stack": stack_test_d,
    "pf": test["pf_d"].to_numpy(float),
    "beam": test["beam_d"].to_numpy(float),
    "selector": (
        0.85 * test["pf_d"].to_numpy(float)
        + 0.15 * test["beam_d"].to_numpy(float)
    ),
    "structural": -test["d_z"].to_numpy(float),
    "hold": np.zeros(len(test), dtype=float),
}
ranker_test_wells, ranker_test_rows, _ = make_well_features(test, candidate_test)
ranker_Xt = np.nan_to_num(
    ranker_test_wells[ranker_feature_cols].to_numpy(np.float32),
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
)
test_prob_raw = ranker_full.predict_proba(ranker_Xt)
test_prob = np.zeros((len(ranker_test_wells), len(candidate_names)), dtype=float)
test_prob[:, ranker_full.classes_.astype(int)] = test_prob_raw
ranker_test_delta = np.zeros(len(test), dtype=float)
weight_rows = []
for row_pos, row in ranker_test_wells.iterrows():
    well = str(row["well"])
    idx = ranker_test_rows[well]
    weight_row = {"well": well, "ranker_accepted": ranker_accepted}
    for candidate_idx, name in enumerate(candidate_names):
        probability = float(test_prob[row_pos, candidate_idx])
        ranker_test_delta[idx] += probability * candidate_test[name][idx]
        weight_row[f"weight_{name}"] = probability
    weight_rows.append(weight_row)
pd.DataFrame(weight_rows).to_csv(WORK / "candidate_ranker_weights.csv", index=False)
if ranker_accepted:
    learned_test_d = np.clip(ranker_test_delta, -STACK_CLIP, STACK_CLIP)
else:
    learned_test_d = stack_test_d
'''
driver_e06 = replace_once(
    driver_e06,
    "learned_test_d = np.clip(\n"
    "    model.predict(Xt) * STACK_SHRINK, -STACK_CLIP, STACK_CLIP\n"
    ")\n"
    "pd.DataFrame(cv_rows).to_csv(WORK / \"training_validation_report.csv\", index=False)",
    "learned_test_d = np.clip(\n"
    "    model.predict(Xt) * STACK_SHRINK, -STACK_CLIP, STACK_CLIP\n"
    ")\n\n"
    + ranker_code.strip()
    + "\n\npd.DataFrame(cv_rows).to_csv(WORK / \"training_validation_report.csv\", index=False)",
)
package(
    "rogii-e06-gpu-candidate-ranker",
    "ROGII E06 GPU Candidate Ranker",
    "# ROGII E06: OOF Candidate Ranker with Safety Gate\n\n" + COMMON_INTRO,
    control_e06,
    driver_e06,
    gpu=True,
)


print("built", len(list(OUT_ROOT.glob("*/kernel-metadata.json"))), "experiment packages in", OUT_ROOT)
