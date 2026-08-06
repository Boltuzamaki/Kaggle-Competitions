# Kaggle submission protocol

Competition: `rogii-wellbore-geology-prediction`

Use this checklist for every future leaderboard attempt. A command that returns
HTTP 4xx before a submission reference is created is a failed request, not a
leaderboard submission, but it must not be retried blindly.

## 1. Build as a competition notebook

The kernel metadata must contain:

```json
{
  "competition_sources": ["rogii-wellbore-geology-prediction"],
  "enable_internet": false
}
```

The notebook must create `/kaggle/working/submission.csv` from the official
competition data. Do not load or blend another notebook's prediction CSV.

## 2. Validate before pushing

- Use five-fold `GroupKFold` grouped by complete well.
- Reproduce the test-time `TVT_input` visible-prefix/hidden-suffix mask.
- Report pooled row-level RMSE, which is the competition metric.
- Confirm every test row is generated once and in sample-submission order.
- Require exactly the columns `id,tvt`, 14,151 unique IDs, no missing values,
  and finite TVT predictions.
- Save the CV score, configuration, row statistics, and SHA-256 hash in an
  audit JSON.

## 3. Push and inspect

Push the private competition kernel once, wait for `COMPLETE`, download its
actual output, and repeat the row/order/finite/hash audit on that downloaded
file. Record the exact owner, slug, and version.

## 4. Submit once

First check:

```bash
kaggle competitions submissions rogii-wellbore-geology-prediction
```

Submit the completed competition-kernel version through Kaggle's code
submission flow. If code submission is unavailable, submit the exact audited
downloaded CSV once. Do not issue a second request unless the first request
returned no submission reference and the rejection reason has been resolved.

After submission, verify that a new reference appears in the submissions list.
Only a created reference consumes a daily slot. Poll that reference for its
score rather than resubmitting.

## Harshini reference calibration

`luffyh04/harshini-submission-f` is a clean reference implementation:

- Inputs: official competition data only
- Validation: five-fold grouped by well, pooled RMSE
- CV wells/rows: 765 / 3,746,966
- Best stacked CV: 8.710
- Public leaderboard: 7.860
- Public minus CV: -0.850 ft (public is 9.76% better)

Its final stack uses a spatial formation model, multi-scale particle filters,
two LightGBM models, CatBoost, a positive Ridge stack, and degree-four robust
structural projection.
