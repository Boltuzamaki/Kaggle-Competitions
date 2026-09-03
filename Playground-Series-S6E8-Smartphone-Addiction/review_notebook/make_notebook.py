"""Generate the private, attribution-first Kaggle review notebook."""

from pathlib import Path
import nbformat as nbf


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kaggle": {"accelerator": "none", "dataSources": [], "dockerImageVersionId": None},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

cells = []
cells.append(nbf.v4.new_markdown_cell("""# S6E8 — Missingness-Aware Ensemble

Public ROC AUC: **0.97069**

Credits:

- [Rayk Kretzschmar — missingness-aware 55-model blend](https://www.kaggle.com/code/raykkretzschmar/s6e8-missingness-aware-55-model-blend)
- [Ripon C. Malo — honest 55-model stack](https://www.kaggle.com/code/riponce/1-public-lb-0-97068-honest-55-model-stack)
- [Naji — S6E8 ensemble](https://www.kaggle.com/code/najiama/s6e8-addiction-lb-0-97062)
- [Szymon Kłapiński — public OOF model library](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models)
"""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TARGET = 'addicted_label'
COMPLETE_WEIGHT = 0.10
INCOMPLETE_WEIGHT = 0.40

def locate(filename, required_text=None):
    roots = [Path('/kaggle/input'), Path('.')]
    hits = []
    for root in roots:
        if root.exists():
            hits.extend(root.rglob(filename))
    if required_text:
        hits = [p for p in hits if required_text.lower() in str(p).lower()]
    if not hits:
        raise FileNotFoundError(f'Cannot find {filename!r} with marker {required_text!r}')
    return min(hits, key=lambda p: (len(p.parts), len(str(p))))

def id_fingerprint(ids):
    payload = np.asarray(ids, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]

train_path = locate('train.csv', 'playground-series-s6e8')
test_path = train_path.with_name('test.csv')
sample_path = train_path.with_name('sample_submission.csv')
anchor_path = locate('submission.csv', '1-public-lb-0-97068-honest-55-model-stack')
residual_path = locate('10_blend_submission.csv', 'predicting-smartphone-addiction-oof-submission-csv')

print('Competition:', train_path.parent)
print('Anchor:', anchor_path)
print('Residual:', residual_path)"""))
cells.append(nbf.v4.new_markdown_cell("## Data audit"))
cells.append(nbf.v4.new_code_cell("""train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)
features = [c for c in test.columns if c != 'id']

assert train['id'].is_unique and test['id'].is_unique
assert test['id'].equals(sample['id'])
assert TARGET in train and TARGET in sample
assert set(train[TARGET].unique()) <= {0, 1}

audit = pd.DataFrame({
    'split': ['train', 'test'],
    'rows': [len(train), len(test)],
    'columns': [train.shape[1], test.shape[1]],
    'mean missing fields': [train[features].isna().sum(1).mean(), test[features].isna().sum(1).mean()],
    'complete rows': [train[features].notna().all(1).sum(), test[features].notna().all(1).sum()],
    'ID fingerprint': [id_fingerprint(train.id), id_fingerprint(test.id)],
})
display(audit.style.format({'rows':'{:,}', 'mean missing fields':'{:.3f}', 'complete rows':'{:,}'}))
print(f'Target positive rate: {train[TARGET].mean():.6f}')"""))
cells.append(nbf.v4.new_markdown_cell("## Prediction alignment"))
cells.append(nbf.v4.new_code_cell("""def aligned_predictions(path, ids):
    frame = pd.read_csv(path)
    assert list(frame.columns) == ['id', TARGET], (path, frame.columns.tolist())
    assert frame.id.is_unique and len(frame) == len(ids)
    out = frame.set_index('id').reindex(ids)[TARGET]
    assert out.notna().all() and np.isfinite(out).all()
    return pd.Series(out.to_numpy(), index=ids.index)

anchor_raw = aligned_predictions(anchor_path, test.id)
residual_raw = aligned_predictions(residual_path, test.id)
anchor_rank = anchor_raw.rank(method='average', pct=True)
residual_rank = residual_raw.rank(method='average', pct=True)

source_audit = pd.DataFrame({
    'source': ['Ripon 55-model stack', 'Naji final blend'],
    'minimum': [anchor_raw.min(), residual_raw.min()],
    'mean': [anchor_raw.mean(), residual_raw.mean()],
    'maximum': [anchor_raw.max(), residual_raw.max()],
    'unique': [anchor_raw.nunique(), residual_raw.nunique()],
})
display(source_audit.style.format({'minimum':'{:.6f}','mean':'{:.6f}','maximum':'{:.6f}','unique':'{:,}'}))
rho = spearmanr(anchor_raw, residual_raw).statistic
print(f'Spearman correlation: {rho:.8f}')
print(f'Mean absolute rank disagreement: {(anchor_rank-residual_rank).abs().mean():.8f}')"""))
cells.append(nbf.v4.new_markdown_cell("## Conditional rank blend"))
cells.append(nbf.v4.new_code_cell("""missing_count = test[features].isna().sum(axis=1)
weights = pd.Series(np.where(missing_count.eq(0), COMPLETE_WEIGHT, INCOMPLETE_WEIGHT), index=test.index)
prediction = (1 - weights) * anchor_rank + weights * residual_rank
submission = pd.DataFrame({'id': test.id, TARGET: prediction})

assert submission.id.equals(sample.id)
assert len(submission) == 296_302
assert submission[TARGET].notna().all()
assert np.isfinite(submission[TARGET]).all()
assert submission[TARGET].between(0, 1).all()
assert submission[TARGET].nunique() > 280_000

regime_report = pd.DataFrame({
    'missingness regime': ['complete', '1–3 missing', '4+ missing'],
    'rows': [(missing_count==0).sum(), missing_count.between(1,3).sum(), (missing_count>=4).sum()],
    'residual weight': [COMPLETE_WEIGHT, INCOMPLETE_WEIGHT, INCOMPLETE_WEIGHT],
})
display(regime_report.style.format({'rows':'{:,}', 'residual weight':'{:.0%}'}))
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission.head()"""))
cells.append(nbf.v4.new_markdown_cell("## Methodology manifest"))
cells.append(nbf.v4.new_code_cell("""manifest = {
    'artifact': 'submission.csv',
    'method_owner': 'Rayk Kretzschmar',
    'anchor_owner': 'Ripon C. Malo',
    'residual_owner': 'Naji',
    'workflow': 'attributed reproduction with alignment and missingness diagnostics',
    'reported_public_auc': 0.97069,
    'test_id_fingerprint': id_fingerprint(test.id),
    'rows': len(submission),
    'weights': {'complete': COMPLETE_WEIGHT, 'incomplete': INCOMPLETE_WEIGHT},
}
Path('/kaggle/working/methodology.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))"""))

nb["cells"] = cells
Path("review_notebook").mkdir(exist_ok=True)
nbf.write(nb, "review_notebook/s6e8_attributed_reproduction_audit.ipynb")
