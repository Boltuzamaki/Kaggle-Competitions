# Kaggriculture recurrent population self-play

This is an experimental learning track; it does not replace `main.py`. The current
champion remains the safety baseline until a checkpoint passes paired evaluation.

## Design

- 24-channel board encoder plus global economy, inventory, shop, and unit features.
- GRU recurrent state for hidden opponent inventory and multi-day context.
- Shared masked action scorer for up to 12 field units.
- Autoregressive 10-slot market head with compact quantity buckets.
- Actor-critic value head trained on terminal win/loss/tie only.
- CPU rollout processes, one GPU PPO learner, and frozen historical checkpoints.
- Training samples actions; evaluation greedily selects the highest-scoring action.

The joint action is factorized because enumerating every combination of 12 unit
actions and 10 market orders is intractable. Invalid actions are conservatively
masked; the official engine remains the final legality check.

## Local smoke test

Install PyTorch into the project environment, then run:

```bash
.venv/bin/pip install torch
.venv/bin/python -m rl.train --smoke --device cpu
.venv/bin/python -m rl.evaluate rl_runs/checkpoints/latest.pt --opponents main.py --seeds 1 --steps 48
```

## Kaggle training

```bash
.venv/bin/python build_rl_notebook.py
kaggle kernels push -p kaggle_rl
kaggle kernels status boltuzamaki/kaggriculture-rl-training
kaggle kernels output boltuzamaki/kaggriculture-rl-training -p research/kaggle_out/rl
```

The current notebook uses three CPU rollout actors and a CPU learner. Kaggle's
assigned P100 is incompatible with its current preinstalled PyTorch build; the
small learner is not yet the throughput bottleneck, so this avoids wasting GPU quota.
Kaggle session limits
can terminate a 2,000-update run; `--resume checkpoint.pt` continues a lineage.
For durable multi-session training, publish checkpoint outputs as a private dataset
and attach that dataset to the next training notebook version.

## Evaluation contract

A learned policy is not strong because its training score rises. It must be evaluated:

1. both seats on identical seeds versus `main.py`;
2. against every full-gauntlet opponent;
3. with no non-DONE episode;
4. with a positive paired confidence interval before replacing the champion.

Use `python -m rl.evaluate` for checkpoint screening. A later export/promotion step
should be added only after a checkpoint first reaches at least 0.50 versus v9.
