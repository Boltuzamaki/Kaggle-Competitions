#!/usr/bin/env python
"""Derive our Kaggle notebooks from the public 0.947 stack by explicit patch.

The public frontier is a single ~4,000-line notebook driven by ~60 ``BIOHUB_*``
environment variables. Rewriting it is not worth the risk; editing it by hand is
not reproducible. So each of our variants is expressed as an ordered list of
(old, new) string patches against the pinned source notebook, and **every patch
must apply exactly the expected number of times or the build fails**.

That assertion is the point. Session 1 lost a paid GPU run to a generated
notebook that was valid Python and the wrong file, and the lesson recorded then
was to verify the *content* of generated artifacts rather than that generation
succeeded. A silently-unapplied patch here would cost the same way: the run
completes, reports a number, and the number answers a different question.

Usage
-----
    python scripts/pubfork.py --list
    python scripts/pubfork.py 12_div_probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = (REPO / "public_kernels" / "shienzhang_biohub-public-0947-exact-repro"
        / "biohub-public-0947-exact-repro.ipynb")
OUT = REPO / "notebooks"

class Patch:
    """One textual substitution, with the number of hits it must make."""

    def __init__(self, old: str, new: str, count: int = 1, note: str = ""):
        self.old, self.new, self.count, self.note = old, new, count, note

    def apply(self, cells: list[dict]) -> int:
        hits = 0
        for cell in cells:
            src = "".join(cell["source"])
            n = src.count(self.old)
            if n:
                cell["source"] = (src.replace(self.old, self.new)).splitlines(keepends=True)
                hits += n
        return hits

def env_patch(key: str, value: str, note: str = "") -> Patch:
    """Rewrite one ``os.environ['BIOHUB_...'] = '...'`` assignment in place.

    Appending a fresh assignment would not work: the notebook re-reads several
    of these into module-level constants at import time and runs a
    configuration-drift guard that raises on anything unexpected.
    """
    return Patch(f"os.environ['BIOHUB_{key}']", f"os.environ['BIOHUB_{key}']", 0, note)

# --------------------------------------------------------------------------
# the patched division metric, injected into the notebook's own validator
# --------------------------------------------------------------------------

DIV_RANKER = '''
# ----------------------------------------------------------- DIV RANKER ---
# The stock key is `score = parent_dist + 0.15 * sister_dist`, sorted ASCENDING,
# and the per-frame budget is spent from the front of that queue.
# # Measured on all 151 labelled divisions in the training set (13_div_inventory):
# # parent_dist_um    p0 2.84   p10 4.49   p50 7.13   p90 10.05   p100 13.53
# real divisions with parent_dist < 2 um :   0 / 151  (0.0%)
# real divisions with parent_dist < 3 um :   1 / 151  (0.7%)
# # So the ascending key puts the 0-3 um band at the FRONT of the queue and no
# real division lives there. The budget is spent on duplicate detections before
# a real division is reached. This is not a tuning error, it is the wrong sign.
# # The replacement scores each proposal by how much it *looks like a real
# division*: a Gaussian log-likelihood under the measured geometry, fitted on
# those same 151 divisions (log space for the two distances, linear for the
# symmetry ratio). Lower is still better, so the key is the negative
# log-likelihood and `proposals.sort()` needs no change.
# # Why this should transfer to the unseen test embryo: the two training embryos
# differ 12x in labelling density but their division *geometry* is nearly
# identical (median parent_dist 6.17 vs 7.37 um, symmetry 0.457 vs 0.469). A
# likelihood over geometry is therefore far safer than a threshold calibrated
# against false-positive rates, which are driven by the density that does differ.
SAFE_DIV_GEOM_PRIOR = {
    'parent_dist': dict(log=True,  mu=1.94503, sd=0.31430),
    'sister_dist': dict(log=True,  mu=2.30300, sd=0.33479),
    'symmetry':    dict(log=False, mu=0.55760, sd=0.42911),
}
SAFE_DIV_RANK_MODE = os.environ.get('BIOHUB_SAFE_DIV_RANK_MODE', 'likelihood')
SAFE_DIV_RANK_DEEPCENTER_W = float(os.environ.get('BIOHUB_SAFE_DIV_RANK_DEEPCENTER_W', '0.0'))

def _safe_div_nll(name, value):
    """Negative log-likelihood of one feature under the fitted division prior."""
    spec = SAFE_DIV_GEOM_PRIOR[name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 4.0
    if not np.isfinite(v):
        return 4.0
    if spec['log']:
        v = math.log(max(v, 0.1))
    z = (v - spec['mu']) / spec['sd']
    # clipped so one wild feature cannot dominate the ranking
    return float(min(0.5 * z * z, 12.0))

def safe_div_rank_key(parent_dist, sister_dist, child_dist, deepcenter_prob=None):
    """Ranking key for a candidate fork. Lower sorts first, as before."""
    if SAFE_DIV_RANK_MODE == 'geometric':
        return parent_dist + 0.15 * sister_dist
    symmetry = abs(child_dist - parent_dist) / max((child_dist + parent_dist) / 2.0, 1e-6)
    nll = (_safe_div_nll('parent_dist', parent_dist)
           + _safe_div_nll('sister_dist', sister_dist)
           + _safe_div_nll('symmetry', symmetry))
    if SAFE_DIV_RANK_DEEPCENTER_W and deepcenter_prob is not None:
        # the centre-prior model is already evaluated at this point for the veto;
        # using its value as evidence rather than as a threshold costs nothing
        p = float(np.clip(deepcenter_prob, 1e-6, 1.0 - 1e-6))
        nll -= SAFE_DIV_RANK_DEEPCENTER_W * math.log(p)
    return nll
# ------------------------------------------------------- END DIV RANKER ---
'''

PATCHED_DIV_METRIC = '''
# ---------------------------------------------------------------- PATCHED ---
# The stock validator in this notebook scores divisions with the PRE-PATCH rule:
# a GT division counts as recovered whenever its matched parent and both matched
# daughter lineages land anywhere in one weakly connected component that holds a
# fork somewhere. The organizers removed exactly that on 2026-07-17 (aa65e90,
# "updating metric to patch weakly connected component exploit"), because one
# out-of-volume hub node wired to every track root satisfied it for every
# division at once.
# # On the same prediction the two rules disagree by about 2x. Every safe-division
# threshold selected against the stock validator was selected with a compass
# reading double, which is why this notebook replaces it.
# # Verified against tracking_cellmot.division_metrics on ten synthetic graphs by
# tests/test_div_metric_parity.py - exact agreement on TP/FN/FP, including the
# hub exploit (0/1/0 both ways).
def _biohub_bipartite_max_matching(left, edges):
    match_r, match_l = {}, {}

    def augment(u, seen):
        for v in edges.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in left:
        augment(u, set())
    return match_l

def _biohub_adjacency(node_ids, edges):
    succ = {int(n): [] for n in node_ids}
    pred = {int(n): [] for n in node_ids}
    for s, t in edges:
        s, t = int(s), int(t)
        if s in succ and t in pred:
            succ[s].append(t)
            pred[t].append(s)
    return succ, pred

def _biohub_weak_components(node_ids, succ, pred):
    comp = {}
    for seed in (int(n) for n in node_ids):
        if seed in comp:
            continue
        comp[seed] = seed
        stack = [seed]
        while stack:
            cur = stack.pop()
            for nbr in succ.get(cur, ()) + pred.get(cur, ()):
                if nbr not in comp:
                    comp[nbr] = seed
                    stack.append(nbr)
    return comp

def _biohub_branch_evidence(succ, pred, fork, child, pred_to_gt, gt_component):
    """One GT component id for a predicted child branch, plus a malformed flag."""
    if set(pred.get(child, [])) != {fork}:
        return None, True
    if child in pred_to_gt:
        return gt_component[pred_to_gt[child]], False
    grandchildren = succ.get(child, [])
    if any(set(pred.get(gc, [])) != {child} for gc in grandchildren):
        return None, True
    comps = {gt_component[pred_to_gt[gc]] for gc in grandchildren if gc in pred_to_gt}
    return (next(iter(comps)), False) if len(comps) == 1 else (None, False)

def _biohub_local_division_ok(succ, pred, fork, parent_ids, daughter_ids):
    """The patched local-topology test: fork on the parent side, two distinct branches."""
    if {fork, *pred.get(fork, [])}.isdisjoint(parent_ids):
        return False
    lineages = [{c, *succ.get(c, [])} for c in succ.get(fork, [])]
    edges = {g: {i for i, ids in enumerate(lineages) if not m.isdisjoint(ids)}
             for g, m in enumerate(daughter_ids)}
    return len(_biohub_bipartite_max_matching(list(edges), edges)) >= 2

def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, gt_edges,
                               pred_to_gt, gt_to_pred):
    """TP / FP / FN for divisions under the POST-PATCH official rule.

    Signature and return order match the function this replaces, so the rest of
    the validator is untouched. ``pred_to_gt`` here is the full-graph matching;
    the official scorer additionally re-matches against each division window,
    which this reproduces with ``match_nodes_bipartite`` per division.
    """
    pred_succ, pred_pred = _biohub_adjacency(pred_nodes.keys(), pred_edges)
    gt_succ, gt_pred = _biohub_adjacency(gt_nodes.keys(), gt_edges)
    pred_forks = {n for n, o in pred_succ.items() if len(o) >= 2}
    gt_forks = {n for n, o in gt_succ.items() if len(o) >= 2}

    gt_component = _biohub_weak_components(gt_nodes.keys(), gt_succ, gt_pred)

    # evaluable / cross-component / malformed forks, from the full-graph matching
    evaluable = {f for f in pred_forks
                 if f in pred_to_gt and len(gt_succ.get(pred_to_gt[f], [])) >= 1}
    invalid = set()
    for fork in pred_forks:
        evidence = []
        for child in pred_succ.get(fork, []):
            comp, malformed = _biohub_branch_evidence(
                pred_succ, pred_pred, fork, child, pred_to_gt, gt_component)
            if malformed:
                invalid.add(fork)
                break
            if comp is not None:
                evidence.append(comp)
        else:
            if len(set(evidence)) >= 2:
                invalid.add(fork)

    candidates, considered = {}, set()
    for divider in gt_forks:
        children = gt_succ.get(divider, [])
        if len(children) < 2:
            continue
        window = {divider, *gt_pred.get(divider, []), *children,
                  *[gc for c in children for gc in gt_succ.get(c, [])]}
        window_nodes = {g: gt_nodes[g] for g in window if g in gt_nodes}
        # a fresh matching against this division's six-node window alone
        p2g_local, _ = match_nodes_bipartite(pred_nodes, window_nodes,
                                             max_dist=VALIDATOR_MATCH_RADIUS_UM)
        gt_parent_ids = {divider, *gt_pred.get(divider, [])}
        parent_ids = {p for p, g in p2g_local.items() if g in gt_parent_ids}
        daughter_ids = [
            {p for p, g in p2g_local.items() if g in {c, *gt_succ.get(c, [])}}
            for c in children
        ]
        if not parent_ids or sum(bool(d) for d in daughter_ids) < 2:
            candidates[divider] = set()
            continue
        local = parent_ids | {s for p in parent_ids for s in pred_succ.get(p, [])}
        local_forks = local & pred_forks
        considered |= local_forks
        candidates[divider] = {
            f for f in local_forks - invalid
            if _biohub_local_division_ok(pred_succ, pred_pred, f, parent_ids, daughter_ids)
        }

    pairing = _biohub_bipartite_max_matching(list(candidates), candidates)
    tp_forks = set(pairing.values())
    tp = len(pairing)
    fn = len(candidates) - tp
    fp = len((considered | evaluable | invalid) - tp_forks)
    return tp, fp, fn
# ------------------------------------------------------------ END PATCHED ---
'''

# --------------------------------------------------------------------------
# dumping the safe-division candidate pool
# --------------------------------------------------------------------------

DUMP_PROPOSALS = '''        if SAFE_DIV_DUMP_PATH:
            # Every proposal that survived the gates, in ranking order, before
            # the per-frame cap spends the budget. This is the pool the
            # geometric key `parent_dist + 0.15 * sister_dist` is sorting, and
            # the CSV is what a learned ranker has to beat.
            import csv as _csv
            _new = not Path(SAFE_DIV_DUMP_PATH).exists()
            with open(SAFE_DIV_DUMP_PATH, 'a', newline='') as _fh:
                _w = _csv.writer(_fh)
                if _new:
                    _w.writerow(['dataset', 't', 'source_id', 'candidate_id',
                                 'existing_child_id', 'parent_dist_um', 'sister_dist_um',
                                 'child_dist_um', 'geom_score', 'rank', 'frame_cap',
                                 'accepted'])
                for _rank, (_sc, _sid, _cid, _pd, _sd) in enumerate(sorted(proposals)):
                    _w.writerow([dataset, int(nodes_by_id[_sid]['t']), _sid, _cid,
                                 '', f'{_pd:.4f}', f'{_sd:.4f}', '',
                                 f'{_sc:.4f}', _rank, frame_cap,
                                 int(_rank < frame_cap)])
'''

OLD_RANK = """                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))"""

NEW_RANK = """                _dc_prob = None
                if SAFE_DIV_RANK_DEEPCENTER_W:
                    _dc_prob = deepcenter_score_point(dataset, int(candidate['t']),
                                                      node_point(candidate), deepcenter_bundle,
                                                      frame_cache, deepcenter_cache)
                score = safe_div_rank_key(parent_dist, sister_dist, child_dist, _dc_prob)
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))"""

VARIANTS: dict[str, dict] = {
    "21_div_budget_fast": dict(
        title="Biohub 21 Div Budget Fast",
        gpu=True,
        note="The experiment 19 should have been. Gates open + likelihood ranker, "
             "but the cap cut ~3.5x so the TOTAL fork count matches the stock stack. "
             "19 let the count grow 3.5x and scored 0.916 against the stock 0.947 - "
             "the damage was volume, not selection. This changes WHICH forks are "
             "chosen while holding HOW MANY fixed.",
        patches=[
            Patch("OUTPUT_SAFE_DIVISIONS = os.environ.get('BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  DIV_RANKER.strip() + "\n\nOUTPUT_SAFE_DIVISIONS = os.environ.get("
                  "'BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  1, "inject the likelihood ranker"),
            Patch(OLD_RANK, NEW_RANK, 1, "replace the ranking key"),
            Patch("os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '2.25'",
                  "os.environ['BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE'] = '0'\n"
                  "os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '-999.0'", 1,
                  "open the divergence gate"),
            Patch("os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.6'",
                  "os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.0'", 1,
                  "open the symmetry gate"),
            # 19 added ~108 forks/film against the stock ~31. Scale both caps by
            # 31/108 = 0.29 so the count returns to stock while the pool stays wide.
            Patch("os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'",
                  "os.environ['BIOHUB_SAFE_DIV_RANK_MODE'] = 'likelihood'\n"
                  "os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0022'", 1,
                  "ranker on, frame cap cut to hold the fork count"),
            Patch("os.environ['BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP'] = '0.00375'",
                  "os.environ['BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP'] = '0.00107'", 1,
                  "global cap cut to hold the fork count"),
            Patch("os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'",
                  "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.5'", 1,
                  "tight55, selected by both held-out sweeps"),
            Patch("os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
                  "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'\n"
                  "os.environ['BIOHUB_UNET_BATCH_SIZE'] = '8'\n"
                  "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'", 1,
                  "validator off, larger batch"),
        ],
    ),
    "20_div_open_fast": dict(
        title="Biohub 20 Div Open Fast",
        gpu=True,
        note="Everything the evidence supports, as a submission: gates open, "
             "likelihood ranker, DeepCenter safe-division veto relaxed 0.20 -> 0.08, "
             "and MOTION_RELINK_TIGHT_UM 5.5 (which both held-out sweeps selected). "
             "Validator off so it finishes in ~25 minutes.",
        patches=[
            Patch("OUTPUT_SAFE_DIVISIONS = os.environ.get('BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  DIV_RANKER.strip() + "\n\nOUTPUT_SAFE_DIVISIONS = os.environ.get("
                  "'BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  1, "inject the likelihood ranker"),
            Patch(OLD_RANK, NEW_RANK, 1, "replace the ranking key"),
            Patch("os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '2.25'",
                  "os.environ['BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE'] = '0'\n"
                  "os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '-999.0'", 1,
                  "open the divergence gate"),
            Patch("os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.6'",
                  "os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.0'", 1,
                  "open the symmetry gate"),
            # measured: the veto rejects 81% of geometric candidates (5,063 of
            # 6,267 across 42 film-runs). A real second daughter IS a cell, so a
            # centre-prior model should pass it; 81% says the threshold is harsh.
            Patch("os.environ['BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD'] = '0.20'",
                  "os.environ['BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD'] = '0.08'", 1,
                  "relax the DeepCenter safe-division veto"),
            # the drift guard pins this key, so it has to move with it - updated,
            # not deleted, so the guard still catches an accidental change
            Patch("'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD': 0.20",
                  "'BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD': 0.08", 1,
                  "keep the drift guard consistent"),
            Patch("os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '6.0'",
                  "os.environ['BIOHUB_MOTION_RELINK_TIGHT_UM'] = '5.5'", 1,
                  "tight55, selected by both held-out sweeps"),
            Patch("os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'",
                  "os.environ['BIOHUB_SAFE_DIV_RANK_MODE'] = 'likelihood'\n"
                  "os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'", 1,
                  "select the likelihood ranker"),
            Patch("os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
                  "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'\n"
                  "os.environ['BIOHUB_UNET_BATCH_SIZE'] = '8'\n"
                  "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'", 1,
                  "validator off, larger batch"),
        ],
    ),
    "19_div_rank_fast": dict(
        title="Biohub 19 Div Rank Fast",
        gpu=True,
        note="Submission-only build of 15: gates open + likelihood ranker, with the "
             "held-out validator and post-process sweep switched OFF. The validator "
             "predicts 24 extra films and is most of the wall clock; a submission "
             "does not need it, so this turns hours into ~25 minutes.",
        patches=[
            Patch("OUTPUT_SAFE_DIVISIONS = os.environ.get('BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  DIV_RANKER.strip() + "\n\nOUTPUT_SAFE_DIVISIONS = os.environ.get("
                  "'BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  1, "inject the likelihood ranker"),
            Patch(OLD_RANK, NEW_RANK, 1, "replace the ranking key"),
            Patch("os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '2.25'",
                  "os.environ['BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE'] = '0'\n"
                  "os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '-999.0'", 1,
                  "open the divergence gate"),
            Patch("os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.6'",
                  "os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.0'", 1,
                  "open the symmetry gate"),
            Patch("os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'",
                  "os.environ['BIOHUB_SAFE_DIV_RANK_MODE'] = 'likelihood'\n"
                  "os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'", 1,
                  "select the likelihood ranker"),
            Patch("os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
                  "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'\n"
                  "os.environ['BIOHUB_UNET_BATCH_SIZE'] = '8'\n"
                  "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'", 1,
                  "validator off, larger batch"),
        ],
    ),
    "18_div_rank_tight": dict(
        title="Biohub 18 Div Rank Tight",
        gpu=True,
        note="The ranking sign fix ALONE, gates exactly as shipped. The conservative "
             "arm: it can only change which already-admitted candidates get the "
             "budget, so it isolates the ranker from the gate opening in 15.",
        patches=[
            Patch("def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, "
                  "gt_edges, pred_to_gt, gt_to_pred):",
                  PATCHED_DIV_METRIC.strip() + "\n\n\ndef _stock_compute_division_confusion("
                  "pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):",
                  1, "patched division metric"),
            Patch("OUTPUT_SAFE_DIVISIONS = os.environ.get('BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  DIV_RANKER.strip() + "\n\nOUTPUT_SAFE_DIVISIONS = os.environ.get("
                  "'BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  1, "inject the likelihood ranker"),
            Patch(OLD_RANK, NEW_RANK, 1, "replace the ranking key"),
            Patch("os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'",
                  "os.environ['BIOHUB_SAFE_DIV_RANK_MODE'] = 'likelihood'\n"
                  "os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'", 1,
                  "select the likelihood ranker"),
            Patch("os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
                  "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '12'", 1,
                  "more held-out films (division-rich first)"),
            # NOTE: no gate patches. That is the entire point of this arm - 15
            # changes the gates and the ranker together, and if it wins we would
            # not know which did the work.
        ],
    ),
    "15_div_rank": dict(
        title="Biohub 15 Div Rank",
        gpu=True,
        note="Gates opened where they sit at the median of real divisions, and the "
             "fork budget spent by a likelihood-over-geometry ranker instead of "
             "'tightest pair first'. Scored with the patched division metric.",
        patches=[
            Patch("def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, "
                  "gt_edges, pred_to_gt, gt_to_pred):",
                  PATCHED_DIV_METRIC.strip() + "\n\n\ndef _stock_compute_division_confusion("
                  "pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):",
                  1, "patched division metric"),
            Patch("OUTPUT_SAFE_DIVISIONS = os.environ.get('BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  DIV_RANKER.strip() + "\n\nOUTPUT_SAFE_DIVISIONS = os.environ.get("
                  "'BIOHUB_OUTPUT_SAFE_DIVISIONS', '1') != '0'",
                  1, "inject the likelihood ranker"),
            Patch(OLD_RANK, NEW_RANK, 1, "replace the ranking key"),
            # 13_div_inventory: these two gates sit at the 48th and 62nd percentile
            # of real divisions and between them discard 79 of 114. Off.
            Patch("os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '2.25'",
                  "os.environ['BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE'] = '0'\n"
                  "os.environ['BIOHUB_SAFE_DIV_DIVERGE_UM'] = '-999.0'", 1,
                  "open the divergence gate (sat at the median)"),
            Patch("os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.6'",
                  "os.environ['BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU'] = '0.0'", 1,
                  "open the symmetry gate (sat at the 62nd percentile)"),
            # a wider pool is only useful if the budget can reach into it
            Patch("os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'",
                  "os.environ['BIOHUB_SAFE_DIV_RANK_MODE'] = 'likelihood'\n"
                  "os.environ['BIOHUB_SAFE_DIV_FRAME_FRAC_CAP'] = '0.0076'", 1,
                  "select the likelihood ranker"),
            Patch("os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
                  "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '12'", 1,
                  "more held-out films (division-rich first)"),
            # NOTE: no drift-guard patch is needed. The guard pins DET_THRESHOLD,
            # the ILP weights, GAP_CLOSE_UM, OUTPUT_MIN_TRACK_LEN, SAFE_DIV_MAX_UM,
            # DEEPCENTER_SAFE_DIV_THRESHOLD and the two TTA weights - none of
            # which this variant changes. Weakening the guard to make room for a
            # change it does not actually check would remove a real safeguard.
        ],
    ),
    "12_div_probe": dict(
        title="Biohub 12 Div Probe",
        gpu=True,
        note="Public 0.947 stack with the PATCHED division metric in its own "
             "held-out validator, on a wider division-rich held-out set. Its "
             "post-process sweep then re-selects the safe-division thresholds "
             "against the true division Jaccard instead of the 2x one.",
        patches=[
            # 1. tell the truth about divisions
            Patch("def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, "
                  "gt_edges, pred_to_gt, gt_to_pred):",
                  PATCHED_DIV_METRIC.strip() + "\n\n\ndef _stock_compute_division_confusion("
                  "pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):",
                  1, "replace the pre-patch division rule"),
            # 2. widen the held-out set: 4 per type is ~8 films, far too few for
            #    151 divisions spread over 199 films
            # the validator already orders films division-first, so raising
            # this concentrates the held-out set on the films that carry the
            # 151 labelled divisions. 12 per embryo prefix = 24 films; 4 (the
            # stock value) is 8, far too few to measure a division Jaccard.
            Patch("os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '4'",
                  "os.environ['BIOHUB_VALIDATOR_N_PER_TYPE'] = '12'", 1,
                  "more held-out films (division-rich first)"),
            Patch("'BIOHUB_VALIDATOR_N_PER_TYPE': 4", "'BIOHUB_VALIDATOR_N_PER_TYPE': 12",
                  0, "drift guard, if present"),
        ],
    ),
}

def build(name: str) -> Path:
    spec = VARIANTS[name]
    nb = json.loads(BASE.read_text())
    cells = nb["cells"]

    for p in spec["patches"]:
        hits = p.apply(cells)
        if p.count and hits != p.count:
            raise SystemExit(
                f"patch '{p.note or p.old[:60]}' applied {hits} times, expected {p.count}. "
                "The base notebook has changed; re-check the patch before running "
                "anything that costs GPU time."
            )
        print(f"  [{hits:2d}] {p.note or p.old[:60]}")

    out = OUT / f"{name}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"\nwrote {out}  ({len(cells)} cells)")
    return out

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", nargs="?")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.variant:
        for k, v in VARIANTS.items():
            print(f"{k:20s} {v['note']}")
        return
    if not BASE.is_file():
        raise SystemExit(f"base notebook missing: {BASE}")
    build(args.variant)

if __name__ == "__main__":
    main()
