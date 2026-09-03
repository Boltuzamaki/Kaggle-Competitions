# Ordered support CatBoost

This is a prepared private CPU candidate. It uses five-fold honest OOF,
outer-fit-only target-free support mappings, leave-one-out support for model
training rows, and bounded tuning inside each outer training partition.

It is not another exact-value target-encoding experiment: no label enters feature
construction. It is also distinct from the dual-view CatBoost because exact keys
are not passed as categorical model inputs. Only compact numerical support,
novelty, and hierarchical backoff features accompany the three raw categorical
columns and raw/composition numerics. CatBoost uses ordered boosting throughout.

Smoke test locally with:

```bash
S6E8_SMOKE=1 .venv/bin/python catboost_ordered_support/experiment.py
```

The script creates OOF/test artifacts and metrics, but never creates or submits a
competition submission.
