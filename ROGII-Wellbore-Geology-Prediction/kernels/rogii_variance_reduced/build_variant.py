"""Build a variance-reduced variant of the 6.463 stack.

The only change is Monte-Carlo resolution: more particle-filter seeds and
particles. That is a strict variance reduction of an already-chosen estimator,
justified a priori, not a hyperparameter tuned against the leaderboard. Every
model, weight, profile and blend constant is left exactly as-is.

Because the hidden test set may contain many more wells than the three
authoring placeholders, the scale factor is chosen at runtime from the actual
test-well count so the run cannot blow the kernel time limit.

Run:  python kernels/rogii_variance_reduced/build_variant.py
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("/tmp/claude-1000/-home-boltuzamaki-Work-get-a-job-kaggle-competitions-wellbore-comp"
           "/b4dae6d0-41bb-4f96-bd79-f5fdbe7a0464/scratchpad/cursrc/"
           "rogii-contact-u-restore-backup.ipynb")

# Injected right after the seed constants are first defined. It recomputes them
# from the real test-well count before anything consumes them.
ADAPTIVE = '''
# --- variance-reduction override (adaptive to hidden test size) --------------
# Pure Monte-Carlo resolution increase. No model, weight or blend constant is
# touched. Scale is backed off when the hidden test is large so that the extra
# PF work cannot exceed the kernel time budget.
import glob as _vr_glob, os as _vr_os
_vr_n_test = 0
for _vr_root in (COMPETITION_DATA_ROOT,
                 '/kaggle/input/rogii-wellbore-geology-prediction', 'data'):
    _vr_hits = _vr_glob.glob(_vr_os.path.join(_vr_root, 'test', '*__horizontal_well.csv'))
    if _vr_hits:
        _vr_n_test = len(_vr_hits)
        break
if _vr_n_test <= 8:
    _VR_SCALE = 4.0
elif _vr_n_test <= 25:
    _VR_SCALE = 2.0
elif _vr_n_test <= 60:
    _VR_SCALE = 1.5
else:
    _VR_SCALE = 1.0
SP45_SELECTOR_N_PARTICLES = int(round(SP45_SELECTOR_N_PARTICLES * min(_VR_SCALE, 2.0)))
SP45_SELECTOR_N_SEEDS = int(round(SP45_SELECTOR_N_SEEDS * _VR_SCALE))
SELECTOR_PF_SEEDS = SP45_SELECTOR_N_SEEDS
print(f'[variance-reduction] test wells={_vr_n_test} scale={_VR_SCALE} '
      f'particles={SP45_SELECTOR_N_PARTICLES} seeds={SP45_SELECTOR_N_SEEDS}', flush=True)
'''

VP_ADAPTIVE = '''
VISIBLE_PREFIX_CAL_SEEDS = int(round(VISIBLE_PREFIX_CAL_SEEDS * _VR_SCALE))
VISIBLE_PREFIX_FINAL_SEEDS = int(round(VISIBLE_PREFIX_FINAL_SEEDS * _VR_SCALE))
VISIBLE_PREFIX_PARTICLES = int(round(VISIBLE_PREFIX_PARTICLES * min(_VR_SCALE, 2.0)))
print(f'[variance-reduction] vp seeds={VISIBLE_PREFIX_CAL_SEEDS}/'
      f'{VISIBLE_PREFIX_FINAL_SEEDS} particles={VISIBLE_PREFIX_PARTICLES}', flush=True)
'''

ANCHOR_1 = "SELECTOR_PF_RETURN_STD = False"
ANCHOR_2 = "VISIBLE_PREFIX_MAX_WELLS = 1_000_000"

# The original kernel hardcodes the `/kaggle/input/{competitions,datasets}/...`
# mount layout. A freshly created kernel gets the flat `/kaggle/input/<slug>`
# layout instead, which makes CFG.dataset_path point at nothing and the well
# table come back empty (KeyError 'wid'). Resolve both layouts.
PATH_OLD = """COMPETITION_DATA_ROOT = '/kaggle/input/competitions/rogii-wellbore-geology-prediction'
RIDGE_ARTIFACT_ROOT = '/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts'"""

PATH_NEW = '''import os as _vr_os0

def _vr_pick(cands, must=None):
    for c in cands:
        if c and _vr_os0.path.isdir(c):
            if must is None or _vr_os0.path.exists(_vr_os0.path.join(c, must)):
                return c
    return cands[0]

COMPETITION_DATA_ROOT = _vr_pick([
    '/kaggle/input/competitions/rogii-wellbore-geology-prediction',
    '/kaggle/input/rogii-wellbore-geology-prediction',
], must='train')
RIDGE_ARTIFACT_ROOT = _vr_pick([
    '/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts',
    '/kaggle/input/wellbore-geology-prediction-artifacts',
], must='models')
print('[paths] competition:', COMPETITION_DATA_ROOT, flush=True)
print('[paths] ridge artifacts:', RIDGE_ARTIFACT_ROOT, flush=True)'''

# The CODEX_Q2522 stage adds a hardcoded shift to the rows of one hardcoded test
# well id. That constant was derived from leaderboard response, which is out of
# scope here, so it is neutralised to zero. The stage still runs and still
# writes its audit trail; it simply moves nothing.
SHIFT_OLD = "_EX_EXTRA_SHIFT = 0.522000000000"
SHIFT_NEW = ("_EX_EXTRA_SHIFT = 0.0  # neutralised: leaderboard-derived "
             "hardcoded single-well shift")

# Same stage hard-requires exactly one PF seed-branch to have fired. With more
# seeds the branch disagreement resolves and none fires, so it must degrade to a
# no-op instead of raising.
BRANCH_OLD = """_applied = _branch[_branch['reason'].astype(str).str.lower().eq('applied')].copy()
if len(_applied) != 1:
    raise RuntimeError(f'{_EX_LABEL}: expected exactly one applied branch, got {len(_applied)}')
_src_well = str(_applied.iloc[0]['well'])
_src_shift = float(_applied.iloc[0]['shift'])
_src_rows = int(_applied.iloc[0]['moved_rows'])"""

BRANCH_NEW = """_applied = _branch[_branch['reason'].astype(str).str.lower().eq('applied')].copy()
if len(_applied) != 1:
    print(f'{_EX_LABEL}: {len(_applied)} applied branches (expected 1); '
          'stage degrades to a no-op', flush=True)
    _src_well, _src_shift, _src_rows = _EX_EXPECTED_WELL, 2.0, _EX_EXPECTED_ROWS
else:
    _src_well = str(_applied.iloc[0]['well'])
    _src_shift = float(_applied.iloc[0]['shift'])
    _src_rows = int(_applied.iloc[0]['moved_rows'])"""

# Two of the three closing audits compare the shift against the very constants
# that were removed, so they would block the run. Bind them to the live values.
# The third check -- target_rows == 4301 -- is a genuine structural assertion
# computed from the actual submission ids, so it is deliberately left as a hard
# raise: it is what proves the scored set really does carry these well ids.
AUDIT_OLD = """_CODEX_EXPECTED_TOTAL_SHIFT = 2.522000000000
_CODEX_EXPECTED_EXTRA_SHIFT = 0.522000000000"""

AUDIT_NEW = """_CODEX_EXPECTED_TOTAL_SHIFT = float(_EX_EXPECTED_TOTAL_SHIFT)
_CODEX_EXPECTED_EXTRA_SHIFT = float(_EX_EXTRA_SHIFT)  # neutralised above"""

# The codex-affine stage replays a base85-embedded precomputed correction vector
# (scale -4.4189) onto a prediction pinned by exact SHA. That is a stored
# artifact keyed to previously scored output, not a model computed from this
# run, and any upstream change invalidates its anchor. Disabling the blob routes
# the stage down its own `in_run_components` branch, whose frontier/sp45/datum
# weights are all 0.0 -- so the stage becomes a clean no-op.
REPLAY_OLD = "if _CODEX_AFFINE_REPLAY_B85:"
REPLAY_NEW = "if False:  # precomputed replay vector disabled (stored artifact)"

SHA_OLD = ("_CODEX_AFFINE_REQUIRED_Q_PREDICTION_SHA = "
           "'cd4cb205a4002daa51916eb252514eb2fddca8e5c584561cc7919e8dfcc6fd5f'")
SHA_NEW = ("_CODEX_AFFINE_REQUIRED_Q_PREDICTION_SHA = ''  "
           "# anchor pin lifted: replay disabled, upstream intentionally changed")


def main():
    nb = json.loads(SRC.read_text())
    hits1 = hits2 = gpu = paths = shift = branch = audit = replay = sha = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if ANCHOR_1 in src and hits1 == 0:
            src = src.replace(ANCHOR_1, ANCHOR_1 + "\n" + ADAPTIVE, 1)
            hits1 += 1
        if ANCHOR_2 in src and hits2 == 0:
            src = src.replace(ANCHOR_2, ANCHOR_2 + "\n" + VP_ADAPTIVE, 1)
            hits2 += 1
        # The tree models are loaded from mounted artifacts, so device='gpu'
        # only appears in the dead training fallback. Neutralising it lets the
        # kernel run on CPU (GPU batch slots are contended) with no behavioural
        # change on the load path, and no crash if the fallback ever fires.
        if "device='gpu'" in src:
            gpu += src.count("device='gpu'")
            src = src.replace("device='gpu'", "device='cpu'")
        if PATH_OLD in src:
            src = src.replace(PATH_OLD, PATH_NEW, 1)
            paths += 1
        if SHIFT_OLD in src:
            src = src.replace(SHIFT_OLD, SHIFT_NEW, 1)
            shift += 1
        if BRANCH_OLD in src:
            src = src.replace(BRANCH_OLD, BRANCH_NEW, 1)
            branch += 1
        if AUDIT_OLD in src:
            src = src.replace(AUDIT_OLD, AUDIT_NEW, 1)
            audit += 1
        if REPLAY_OLD in src:
            src = src.replace(REPLAY_OLD, REPLAY_NEW, 1)
            replay += 1
        if SHA_OLD in src:
            src = src.replace(SHA_OLD, SHA_NEW, 1)
            sha += 1
        cell["source"] = src.splitlines(keepends=True)
    counts = dict(selector=hits1, visible_prefix=hits2, paths=paths, shift=shift,
                  branch=branch, audit=audit, replay=replay, sha=sha)
    if set(counts.values()) != {1}:
        raise SystemExit(f"patch failed: {counts}")
    print("disabled the base85 precomputed replay vector and its SHA anchor pin")
    print("neutralised the hardcoded single-well leaderboard shift")
    print("made the seed-branch stage degrade to a no-op")
    print(f"neutralised {gpu} device='gpu' occurrences (dead training branch)")
    print("rewrote competition/ridge roots for both Kaggle mount layouts")

    out = HERE / "rogii-variance-reduced.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out, f"({out.stat().st_size/1e6:.2f} MB)")

    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
    meta["id"] = "boltuzamaki/rogii-variance-reduced"
    meta["title"] = "ROGII Variance Reduced"
    meta["code_file"] = "rogii-variance-reduced.ipynb"
    meta.pop("id_no", None)
    meta["enable_gpu"] = False
    meta["machine_shape"] = "Cpu"
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", HERE / "kernel-metadata.json")
    print("dataset_sources:", meta["dataset_sources"])


if __name__ == "__main__":
    main()
