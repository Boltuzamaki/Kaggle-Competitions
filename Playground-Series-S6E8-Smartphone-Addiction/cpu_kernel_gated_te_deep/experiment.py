"""Architectures other than a plain MLP, all on the fold-safe target-encoded view.

The ledger rejected four neural architectures outright: FT-Transformer 0.9401,
DCNv2 0.9385, GANDALF 0.9387, TabR 0.9380. Every one of those was trained
target-free, and the recorded conclusion was "do not spend more GPU time on
target-free architectures however novel they are."

That conclusion is about the *view*, not about the architectures. When a plain
three-layer MLP was finally given the explicit fold-safe target encoding it
scored 0.9661121 and survived a paired test at t=+3.1, and the wider variant at
t=+3.1 as well. So the 0.030 deficit belonged to the missing target statistics,
and no architecture other than a fully-connected stack has ever been run on the
view that supplies them.

That is the gap this file closes. Four learners, one encoder:

  resnet_te  pre-norm residual blocks. A plain MLP composes one transformation
             per layer; a residual stack composes corrections to an identity,
             which is a different function class at the same parameter count and
             trains stably far deeper.
  cnn1d_te   the features are projected to a grid and read by 1D convolutions.
             Weight sharing across positions is a genuinely different prior: it
             cannot express an arbitrary per-feature weight, which is exactly why
             its errors should not line up with a dense network's.
  dae_te     a denoising autoencoder pre-trained with swap noise, then a head on
             the bottleneck. The representation is fitted with no labels at all,
             so what it learns is the joint structure of the feature block rather
             than anything about the target, and the supervised head starts from
             a basis no discriminative model would have chosen.
  ftt_te     FT-Transformer, the strongest of the four rejected models, re-run on
             the view it was never given. Attention over feature tokens is the
             one mechanism here that can weight interactions per row instead of
             learning them globally. GPU kernel: 100-odd tokens of self-attention
             over 691k rows is not a CPU job.

The stack does not need any of these to be accurate. It needs them to be wrong
in a different place from the trees, inside the accuracy band the ledger
measured: cpu_extratrees pays at 0.9617, tabm_lattice fails at 0.9592, the RBF
machine fails at 0.9515. Anything here that lands above roughly 0.962 is a
candidate; anything below it is not, however novel.

The encoder below is copied byte-for-byte from cpu_kernel_mlp_te, which took it
from cpu_kernel_lgb10fold. That is deliberate: holding the features fixed is what
makes the comparison between learners mean anything.

Official competition data only. No public predictions and no submission call.
"""
from pathlib import Path
import argparse
import sys
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID = "addicted_label", "id"
SEED, INNER_FOLDS = 20260807, 5
SMOOTHING = 40.0

# One entry per kernel. Fold and seed counts are set from the measured cost of
# the mlp_te family: 5 folds x 2 seeds of a 512-256-128 net took 50 minutes on
# four Kaggle cores, against a 12-hour limit. Ten folds is the validated lever
# (+0.00018 on XGBoost, E042b) so anything cheap enough gets it. The autoencoder
# pays for a second training pass per fold and the transformer is quadratic in
# tokens, so both stay at five.
VARIANTS = {
    "resnet_te": {"folds": 10, "seeds": 2, "epochs": 20, "gpu": False,
                  "arch": "resnet", "d": 256, "hidden": 512, "blocks": 4, "drop": 0.15},
    "cnn1d_te":  {"folds": 10, "seeds": 2, "epochs": 20, "gpu": False,
                  "arch": "cnn1d", "channels": 32, "length": 24, "drop": 0.20},
    "dae_te":    {"folds": 5,  "seeds": 2, "epochs": 20, "gpu": False,
                  "arch": "dae", "bottleneck": 128, "dae_epochs": 8,
                  "swap": 0.15, "drop": 0.15},
    # Attention is quadratic in tokens and there is one token per feature, so the
    # activation for a batch of 8192 over ~120 tokens does not fit in 8GB. The
    # batch is dropped to 1024 for this variant only; every other learner here is
    # linear in features and keeps the large batch.
    #
    # This variant is run on the local RTX 4060, not on Kaggle. Kaggle's P100 is
    # sm_60 and the torch build in the current image raises
    # `no kernel image is available for execution on the device` for it. The code
    # is unchanged; only the card is.
    "ftt_te":    {"folds": 5,  "seeds": 1, "epochs": 12, "gpu": True, "batch": 1024,
                  "arch": "ftt", "d_token": 64, "layers": 3, "heads": 8, "drop": 0.10},
    # Second wave. E060 established that the ledger's neural rejections were
    # about the target-free view rather than the architectures, so the remaining
    # rejected designs are re-run here on the view that works. dae_te also gets
    # the ten-fold lever, which is worth more than any of the tuning tried so far.
    "dcnv2_te":  {"folds": 10, "seeds": 2, "epochs": 20, "gpu": False,
                  "arch": "dcnv2", "d": 384, "cross_depth": 4, "drop": 0.15},
    # Measured 9.6h for 8 of 10 folds, so ten never had a chance inside the
    # 12h limit. Five folds of a complete stream beats ten folds of a partial one
    # the registry will not load.
    "gated_te":  {"folds": 5, "seeds": 2, "epochs": 20, "gpu": False,
                  "arch": "gated", "d": 512, "blocks": 4, "drop": 0.15},
    "dae_te_10f": {"folds": 10, "seeds": 2, "epochs": 20, "gpu": False,
                   "arch": "dae", "bottleneck": 128, "dae_epochs": 8,
                   "swap": 0.15, "drop": 0.15},
    # Only 4 of 10 folds in 8.6h: eight residual blocks at 24 epochs is the most
    # expensive shape tried here. Five folds and a single seed brings it in.
    "resnet_te_deep": {"folds": 5, "seeds": 1, "epochs": 24, "gpu": False,
                       "arch": "resnet", "d": 192, "hidden": 768, "blocks": 8,
                       "drop": 0.20},
    # Third wave, local GPU only. FT-Transformer came back at 0.9671574, the best
    # network in the inventory, so the lever that worked on the lookup family is
    # applied to this one: more folds, and a second shape for a differently-placed
    # error rather than a better score. Five folds took 103 minutes on the 4060.
    "ftt_te_10f":  {"folds": 10, "seeds": 1, "epochs": 12, "gpu": True, "batch": 1024,
                    "arch": "ftt", "d_token": 64, "layers": 3, "heads": 8, "drop": 0.10},
    "ftt_te_wide": {"folds": 5,  "seeds": 1, "epochs": 14, "gpu": True, "batch": 768,
                    "arch": "ftt", "d_token": 96, "layers": 4, "heads": 8, "drop": 0.15},
    # Fourth wave. NODE is the one remaining mechanism in the inventory that is
    # neither a dense net nor a boosted tree: an ensemble of *differentiable*
    # oblivious trees, fitted end to end by gradient descent rather than greedily
    # split by split. The leaf structure is a tree's, the fitting procedure is a
    # network's, so its errors should sit between the two families that dominate
    # the stack. Batch is small because the leaf tensor is trees x leaves wide.
    "node_te":   {"folds": 5, "seeds": 2, "epochs": 16, "gpu": False, "batch": 2048,
                  "arch": "node", "trees": 48, "depth": 5, "tau": 0.5, "drop": 0.0},
    # Cheap shape diversity inside families already proven to be in band. These
    # are not expected to beat their parents; they are expected to be wrong
    # somewhere else, which is the only thing the stack pays for.
    "gated_te_deep":  {"folds": 5, "seeds": 2, "epochs": 22, "gpu": False,
                       "arch": "gated", "d": 384, "blocks": 8, "drop": 0.20},
    "resnet_te_wide": {"folds": 5, "seeds": 2, "epochs": 20, "gpu": False,
                       "arch": "resnet", "d": 512, "hidden": 1024, "blocks": 3,
                       "drop": 0.25},
    "dae_te_big":     {"folds": 5, "seeds": 2, "epochs": 20, "gpu": False,
                       "arch": "dae", "bottleneck": 256, "dae_epochs": 10,
                       "swap": 0.25, "drop": 0.15},
}
VARIANT = "gated_te_deep"  # generated; edit the source, not this copy

CFG = VARIANTS[VARIANT]
OUTER_FOLDS = CFG["folds"]
NET_SEEDS = CFG["seeds"]

PAIR_COLUMNS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("notifications_per_day", "app_opens_per_day"),
    ("daily_screen_time_hours", "gaming_hours"),
    ("sleep_hours", "stress_level"),
]


def locate():
    for root in (Path("/kaggle/input"), Path(".."), Path(".")):
        if not root.exists():
            continue
        for p in root.rglob("train.csv"):
            try:
                cols = pd.read_csv(p, nrows=1).columns
            except Exception:
                continue
            if TARGET in cols and (p.parent / "test.csv").exists():
                return p, p.parent / "test.csv"
    raise FileNotFoundError("official train.csv/test.csv not found")


def base_features(frame):
    x = frame.drop(columns=[ID, TARGET], errors="ignore").copy()
    numeric = list(x.select_dtypes(include="number").columns)
    x["missing_count"] = x.isna().sum(axis=1).astype("int8")
    for column in numeric:
        x[column + "__missing"] = x[column].isna().astype("int8")
    x["leisure_hours"] = x["social_media_hours"] + x["gaming_hours"]
    x["accounted_hours"] = x["leisure_hours"] + x["work_study_hours"]
    x["unaccounted_screen"] = x["daily_screen_time_hours"] - x["accounted_hours"]
    x["weekend_unaccounted"] = x["weekend_screen_time"] - x["accounted_hours"]
    x["weekend_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["weekend_ratio"] = x["weekend_screen_time"] / (x["daily_screen_time_hours"] + 0.25)
    x["leisure_share"] = x["leisure_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["social_share"] = x["social_media_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["gaming_share"] = x["gaming_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["work_share"] = x["work_study_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["notif_per_screen"] = x["notifications_per_day"] / (x["daily_screen_time_hours"] + 0.25)
    x["opens_per_screen"] = x["app_opens_per_day"] / (x["daily_screen_time_hours"] + 0.25)
    x["notif_per_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 2.0)
    x["sleep_screen_balance"] = x["sleep_hours"] - x["daily_screen_time_hours"]
    x["screen_sleep_ratio"] = x["daily_screen_time_hours"] / (x["sleep_hours"] + 0.25)
    for column in x.select_dtypes(exclude="number").columns:
        x[column] = x[column].astype("category").cat.codes.astype("int16")
    # The tree version also carries a `missing_pattern` bitmask. It is dropped
    # here: an integer whose bits are meaningful but whose magnitude is not is
    # readable by an axis-aligned split and pure noise to a smooth network. The
    # per-column flags plus missing_count carry the same information in a form
    # a network can actually use.
    return x.replace([np.inf, -np.inf], np.nan)


def key_of(frame, column, digits=None):
    s = frame[column]
    if digits is not None and pd.api.types.is_numeric_dtype(s):
        s = s.round(digits)
    return s.astype("string").fillna("__NA__")


def pair_key(frame, left, right, digits=None):
    return key_of(frame, left, digits) + "|" + key_of(frame, right, digits)


def rate_map(keys, y, prior):
    stats = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k").y.agg(["sum", "count"])
    return (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)


def inner_folds(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    for k in range(INNER_FOLDS):
        va = order[k::INNER_FOLDS]
        yield np.setdiff1d(order, va, assume_unique=True), va


def build(fit_raw, fit_y, apply_raw, apply_base, raw_columns, inner_oof):
    """Fold-safe encoder, identical contract to cpu_kernel_lgb10fold."""
    prior = float(np.mean(fit_y))
    cols = {}
    for column in raw_columns:
        counts = key_of(fit_raw, column).value_counts()
        cols[column + "__logfreq"] = np.log1p(
            key_of(apply_raw, column).map(counts).fillna(0)).to_numpy("float32")
        if pd.api.types.is_numeric_dtype(fit_raw[column]):
            rc = key_of(fit_raw, column, 1).value_counts()
            cols[column + "__rlogfreq"] = np.log1p(
                key_of(apply_raw, column, 1).map(rc).fillna(0)).to_numpy("float32")
    for left, right in PAIR_COLUMNS:
        for tag, digits in (("exact", None), ("rounded", 1)):
            counts = pair_key(fit_raw, left, right, digits).value_counts()
            cols[f"{left}__{right}__{tag}_lf"] = np.log1p(
                pair_key(apply_raw, left, right, digits).map(counts).fillna(0)).to_numpy("float32")

    specs = [(c,) for c in raw_columns] + [p for p in PAIR_COLUMNS]
    for spec in specs:
        name = "__".join(spec) + "__te"
        fk = key_of(fit_raw, spec[0]) if len(spec) == 1 else pair_key(fit_raw, *spec)
        if inner_oof:
            values = np.full(len(apply_raw), np.nan, dtype="float32")
            for tr_i, va_i in inner_folds(len(fk), SEED):
                r = rate_map(fk.iloc[tr_i], fit_y[tr_i], prior)
                values[va_i] = fk.iloc[va_i].map(r).fillna(prior).to_numpy("float32")
            cols[name] = values
        else:
            ak = key_of(apply_raw, spec[0]) if len(spec) == 1 else pair_key(apply_raw, *spec)
            cols[name] = ak.map(rate_map(fk, fit_y, prior)).fillna(prior).to_numpy("float32")
    return pd.concat([apply_base, pd.DataFrame(cols, index=apply_base.index)], axis=1)


# --- the learners ---------------------------------------------------------


def make_resnet(torch, n_features):
    """Pre-norm residual blocks, the tabular ResNet of Gorishniy et al.

    Each block returns `x + f(x)`, so depth adds corrections to an identity
    rather than replacing the representation. That is what lets this run four
    blocks deep where the plain stack tops out at three layers, and depth is the
    axis along which its errors should stop resembling the plain stack's.
    """
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self, d, hidden, drop):
            super().__init__()
            self.norm = nn.BatchNorm1d(d)
            self.up = nn.Linear(d, hidden)
            self.down = nn.Linear(hidden, d)
            self.drop = nn.Dropout(drop)
            self.act = nn.GELU()

        def forward(self, x):
            z = self.act(self.up(self.norm(x)))
            return x + self.down(self.drop(z))

    d, hidden = CFG["d"], CFG["hidden"]
    return nn.Sequential(
        nn.Linear(n_features, d),
        *[Block(d, hidden, CFG["drop"]) for _ in range(CFG["blocks"])],
        nn.BatchNorm1d(d), nn.GELU(), nn.Linear(d, 1),
    )


def make_cnn1d(torch, n_features):
    """Project to a grid, then read it with shared convolutional filters.

    The projection is learned, so the grid has no inherent order; what matters is
    that every position is then read by the *same* small filter bank. A dense
    layer can give each feature its own weight and this cannot, which is a
    strictly narrower hypothesis class and therefore a different set of mistakes.
    """
    nn = torch.nn
    channels, length = CFG["channels"], CFG["length"]

    class Reshape(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), channels, length)

    return nn.Sequential(
        nn.Linear(n_features, channels * length),
        nn.BatchNorm1d(channels * length), nn.GELU(),
        Reshape(),
        nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2),
        nn.BatchNorm1d(channels * 2), nn.GELU(),
        nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1),
        nn.BatchNorm1d(channels * 2), nn.GELU(),
        nn.AdaptiveAvgPool1d(4),
        nn.Flatten(),
        nn.Dropout(CFG["drop"]),
        nn.Linear(channels * 8, 128), nn.BatchNorm1d(128), nn.GELU(),
        nn.Linear(128, 1),
    )


def make_ftt(torch, n_features):
    """FT-Transformer: one token per feature, self-attention over the tokens.

    Every other learner here decides how to combine two features once, globally.
    Attention decides it per row. That is the mechanism worth paying for, and the
    reason this architecture is worth a second look on a view it has never seen.
    """
    nn = torch.nn
    d, heads, layers, drop = CFG["d_token"], CFG["heads"], CFG["layers"], CFG["drop"]

    class FTT(nn.Module):
        def __init__(self):
            super().__init__()
            # A scalar feature becomes a token by an affine map with per-feature
            # weight and bias, which is the numerical tokenizer from the paper.
            self.weight = nn.Parameter(torch.randn(n_features, d) * d ** -0.5)
            self.bias = nn.Parameter(torch.zeros(n_features, d))
            self.cls = nn.Parameter(torch.randn(1, 1, d) * d ** -0.5)
            enc = nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=d * 2, dropout=drop,
                activation="gelu", batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, 1))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.weight + self.bias
            tok = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], dim=1)
            # Only the CLS token is read out, so the head sees a row-specific
            # summary of the whole feature set rather than any single token.
            return self.head(self.body(tok)[:, 0])

    return FTT()


def make_dcnv2(torch, n_features):
    """Explicit bounded-degree feature crossing beside a deep tower.

    A cross layer computes `x0 * (W h + b) + h`, so after k layers the output
    contains products of at most k+1 raw inputs and nothing higher. Every other
    learner here approximates interactions with a stack of nonlinearities; this
    one constructs them, which bounds what it can express and makes its errors
    structurally unlike a free-form MLP's.
    """
    nn = torch.nn

    class DCNv2(nn.Module):
        def __init__(self):
            super().__init__()
            d, depth = CFG["d"], CFG["cross_depth"]
            self.stem = nn.Sequential(nn.Linear(n_features, d), nn.BatchNorm1d(d), nn.GELU())
            self.cross = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
            self.deep = nn.Sequential(
                nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(CFG["drop"]),
                nn.Linear(d, d // 2), nn.BatchNorm1d(d // 2), nn.GELU(),
            )
            self.head = nn.Linear(d + d // 2, 1)

        def forward(self, x):
            x0 = self.stem(x)
            h = x0
            for layer in self.cross:
                h = x0 * layer(h) + h
            return self.head(torch.cat([h, self.deep(x0)], dim=1))

    return DCNv2()


def make_gated(torch, n_features):
    """Gated linear units: one branch decides how much of the other passes.

    This is the mechanism GANDALF's feature-gating unit is built on, kept here in
    its plain form. A gate is multiplicative where every layer in the plain MLP is
    additive-then-squashed, so it can suppress a feature for one kind of row and
    pass it for another without needing depth to do it.
    """
    nn = torch.nn

    class GatedBlock(nn.Module):
        def __init__(self, d, drop):
            super().__init__()
            self.norm = nn.BatchNorm1d(d)
            self.value = nn.Linear(d, d)
            self.gate = nn.Linear(d, d)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            z = self.norm(x)
            return x + self.drop(self.value(z) * torch.sigmoid(self.gate(z)))

    d = CFG["d"]
    return nn.Sequential(
        nn.Linear(n_features, d),
        *[GatedBlock(d, CFG["drop"]) for _ in range(CFG["blocks"])],
        nn.BatchNorm1d(d), nn.GELU(), nn.Linear(d, 1),
    )


def make_node(torch, n_features):
    """An ensemble of differentiable oblivious decision trees.

    An oblivious tree asks the same question at every node of a level, so a depth
    `D` tree is fully described by `D` splits and `2**D` leaf values. Here each
    split is a learned linear projection through a sigmoid rather than a hard
    threshold on one column, which makes the whole ensemble differentiable and
    fitted jointly by gradient descent.

    That is the interesting part. Every boosted tree in this stack was built
    greedily, one split at a time, each conditioned on the residual left by the
    last. These trees are all fitted at once against the same loss, so the
    ensemble can arrange for its members to be complementary in a way a greedy
    procedure has no opportunity to. Same leaf-structured hypothesis class,
    completely different route to it.
    """
    nn = torch.nn
    trees, depth, tau = CFG["trees"], CFG["depth"], CFG["tau"]
    n_leaves = 2 ** depth

    class ODT(nn.Module):
        def __init__(self):
            super().__init__()
            self.select = nn.Parameter(torch.randn(trees, depth, n_features) * n_features ** -0.5)
            self.threshold = nn.Parameter(torch.zeros(trees, depth))
            self.leaf = nn.Parameter(torch.randn(trees, n_leaves) * 0.1)
            # bits[l, d] is 1 when leaf l lies on the "right" side of split d, so
            # the leaf probability is a product over depth of p or (1 - p).
            bits = ((torch.arange(n_leaves).unsqueeze(1)
                     >> torch.arange(depth).flip(0)) & 1).float()
            self.register_buffer("bits", bits)

        def forward(self, x):
            # (B, T, D): how far each row is past each split
            h = torch.einsum("bf,tdf->btd", x, self.select) - self.threshold
            p = torch.sigmoid(h / tau).unsqueeze(2)          # (B, T, 1, D)
            bits = self.bits.view(1, 1, n_leaves, depth)
            leaf_p = (p * bits + (1.0 - p) * (1.0 - bits)).prod(-1)   # (B, T, L)
            return (leaf_p * self.leaf).sum(dim=(1, 2)).unsqueeze(1)

    return ODT()


def make_net(torch, n_features, seed):
    torch.manual_seed(seed)
    return {"resnet": make_resnet, "cnn1d": make_cnn1d, "ftt": make_ftt,
            "dcnv2": make_dcnv2, "gated": make_gated,
            "node": make_node}[CFG["arch"]](torch, n_features)


def pretrain_dae(torch, a_fit, seed, device, threads):
    """Swap-noise denoising autoencoder over the encoded block, labels unused.

    Swap noise corrupts a cell by replacing it with the value another row holds
    in the same column, so the marginal distribution of every column is left
    exactly as it was and only the dependence between columns is broken. The
    network's job is to restore that dependence, which forces the bottleneck to
    carry the joint structure of the block.

    No label is read here, so this pass cannot leak: it would be sound on the
    test rows too. It is restricted to the fit rows anyway, because a
    representation fitted on rows the outer fold is about to score is a harder
    thing to reason about than one that is not.
    """
    nn = torch.nn
    torch.manual_seed(seed)
    n_features, bottleneck = a_fit.shape[1], CFG["bottleneck"]
    encoder = nn.Sequential(
        nn.Linear(n_features, 512), nn.BatchNorm1d(512), nn.GELU(),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(),
        nn.Linear(256, bottleneck), nn.BatchNorm1d(bottleneck), nn.GELU(),
    ).to(device)
    decoder = nn.Sequential(
        nn.Linear(bottleneck, 256), nn.BatchNorm1d(256), nn.GELU(),
        nn.Linear(256, 512), nn.BatchNorm1d(512), nn.GELU(),
        nn.Linear(512, n_features),
    ).to(device)

    t_fit = torch.from_numpy(a_fit)
    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    batch, swap = 8192, CFG["swap"]

    encoder.train(); decoder.train()
    for epoch in range(CFG["dae_epochs"]):
        order = torch.randperm(len(t_fit), generator=gen)
        total = 0.0
        for b in range(0, len(order), batch):
            clean = t_fit[order[b:b + batch]].to(device)
            if len(clean) < 2:
                continue
            # For each cell, with probability `swap`, take the value another row
            # in this batch holds in the same column.
            donor = clean[torch.randint(len(clean), (len(clean),), generator=gen).to(device)]
            mask = (torch.rand(clean.shape, generator=gen).to(device) < swap)
            noisy = torch.where(mask, donor, clean)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(decoder(encoder(noisy)), clean)
            loss.backward()
            opt.step()
            total += float(loss) * len(clean)
        print(f"      dae epoch {epoch + 1} mse {total / len(t_fit):.5f}", flush=True)

    encoder.eval()
    return encoder


def make_dae_head(torch, n_features, bottleneck, seed):
    """MLP over the original block concatenated with the learned bottleneck."""
    nn = torch.nn
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(n_features + bottleneck, 512), nn.BatchNorm1d(512), nn.GELU(),
        nn.Dropout(CFG["drop"]),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(CFG["drop"]),
        nn.Linear(256, 1),
    )


def fit_learner(x_fit, fit_y, x_val, x_test, fold, threads, epochs, batch=None):
    batch = CFG.get("batch", 8192) if batch is None else batch
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from sklearn.pipeline import make_pipeline

    torch.set_num_threads(threads)
    device = torch.device("cuda" if (CFG["gpu"] and torch.cuda.is_available()) else "cpu")
    prep = make_pipeline(
        SimpleImputer(strategy="median"),
        QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                            subsample=200000, random_state=SEED + fold),
        StandardScaler(),
    )
    a_fit = prep.fit_transform(x_fit).astype("float32")
    a_val = prep.transform(x_val).astype("float32")
    a_test = prep.transform(x_test).astype("float32")

    t_fit = torch.from_numpy(a_fit)
    t_y = torch.from_numpy(fit_y.astype("float32")).unsqueeze(1)
    t_val = torch.from_numpy(a_val)
    t_test = torch.from_numpy(a_test)

    val_acc = np.zeros(len(a_val), dtype="float64")
    test_acc = np.zeros(len(a_test), dtype="float64")
    seed_started = time.time()

    for s in range(NET_SEEDS):
        seed_s = SEED + 101 * fold + s
        encoder = None
        if CFG["arch"] == "dae":
            encoder = pretrain_dae(torch, a_fit, seed_s, device, threads)
            net = make_dae_head(torch, a_fit.shape[1], CFG["bottleneck"], seed_s).to(device)
            params = list(net.parameters())
        else:
            net = make_net(torch, a_fit.shape[1], seed_s).to(device)
            params = list(net.parameters())

        def forward(batch_x):
            """One place where the DAE's concatenation happens, so the training
            and inference paths cannot drift apart."""
            if encoder is None:
                return net(batch_x)
            with torch.no_grad():
                z = encoder(batch_x)
            return net(torch.cat([batch_x, z], dim=1))

        opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-5)
        n_batches = (len(t_fit) + batch - 1) // batch
        # OneCycleLR divides by the length of its annealing phase, which rounds
        # to zero when there are only a handful of steps. That never happens on
        # the full data but does on a row-capped smoke run, so the floor keeps
        # the smoke path exercising the real code.
        total_steps = max(epochs * n_batches, 16)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=total_steps, pct_start=0.25)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(SEED + 7919 * fold + s)

        net.train()
        for epoch in range(epochs):
            order = torch.randperm(len(t_fit), generator=gen)
            for b in range(n_batches):
                idx = order[b * batch:(b + 1) * batch]
                if len(idx) < 2:  # BatchNorm needs more than one row
                    continue
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(forward(t_fit[idx].to(device)), t_y[idx].to(device))
                loss.backward()
                opt.step()
                sched.step()

        net.eval()
        with torch.no_grad():
            for source, sink in ((t_val, val_acc), (t_test, test_acc)):
                chunks = [torch.sigmoid(forward(source[i:i + 16384].to(device))).squeeze(1).cpu().numpy()
                          for i in range(0, len(source), 16384)]
                sink += np.concatenate(chunks)
        # No validation label reaches this function, so there is no early
        # stopping and nothing to select: the seed loop just reports progress.
        print(f"    fold {fold} net seed {s} trained ({time.time() - seed_started:.0f}s)",
              flush=True)
        del net, encoder
        gc.collect()

    return val_acc / NET_SEEDS, test_acc / NET_SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--folds", type=int, default=OUTER_FOLDS)
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--rows", type=int, default=0, help="smoke test row cap")
    ap.add_argument("--total-budget", type=float, default=10.5 * 3600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_folds = args.folds
    out_dir = Path(args.out) if args.out else (
        Path("/kaggle/working") if Path("/kaggle").exists()
        else Path(__file__).parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    train_path, test_path = locate()
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if args.rows:
        # Stratified cap by positional index. `groupby(...).apply(...)` is not
        # used because whether it keeps the grouping column differs between the
        # pandas 2 that Kaggle runs and the pandas 3 installed locally.
        per_class = max(1, args.rows // 2)
        take = np.concatenate([
            np.flatnonzero(train[TARGET].to_numpy() == c)[:per_class] for c in (0, 1)])
        train = train.iloc[np.sort(take)].reset_index(drop=True)
        test = test.head(min(len(test), args.rows // 2)).reset_index(drop=True)

    raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
    y = train[TARGET].to_numpy("int8")
    train_base, test_base = base_features(train), base_features(test)
    train_raw, test_raw = train[raw_columns], test[raw_columns]

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    records = []
    folds_done = 0
    outer = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(train_base, y))

    for fold, (fit_i, val_i) in enumerate(outer_splits, 1):
        # A fold that cannot finish inside the kernel's wall clock would be
        # killed mid-write, so the loop stops while there is still time to
        # emit artifacts for the folds that did complete.
        remaining = args.total_budget - (time.time() - started)
        if folds_done and remaining < (time.time() - started) / folds_done:
            print(f"[{VARIANT}] stopping before fold {fold}: {remaining:.0f}s left", flush=True)
            break

        fit_raw, fit_y = train_raw.iloc[fit_i], y[fit_i]
        x_fit = build(fit_raw, fit_y, fit_raw, train_base.iloc[fit_i], raw_columns, True)
        x_val = build(fit_raw, fit_y, train_raw.iloc[val_i], train_base.iloc[val_i],
                      raw_columns, False)[x_fit.columns]
        x_test = build(fit_raw, fit_y, test_raw, test_base, raw_columns, False)[x_fit.columns]

        val_pred, test_part = fit_learner(x_fit, fit_y, x_val, x_test, fold,
                                          args.threads, args.epochs)
        oof[val_i] = val_pred
        test_pred += test_part
        folds_done += 1
        auc = roc_auc_score(y[val_i], val_pred)
        records.append({"fold": fold, "outer_auc": float(auc)})
        print(f"[{VARIANT}] fold {fold} AUC {auc:.8f} ({time.time() - started:.0f}s)", flush=True)
        del x_fit, x_val, x_test
        gc.collect()

    if folds_done == 0:
        raise RuntimeError("no outer fold completed; nothing to write")

    complete = folds_done == n_folds
    test_pred /= folds_done
    scored_rows = np.concatenate([outer_splits[i][1] for i in range(folds_done)])
    pooled = float(roc_auc_score(y[scored_rows], oof[scored_rows]))
    print(f"[{VARIANT}] pooled OOF AUC {pooled:.10f} over {folds_done}/{n_folds} folds",
          flush=True)

    if complete:
        pd.DataFrame({ID: train[ID], "fold": -1, "y": y, "pred": oof}).to_csv(
            out_dir / f"oof_{VARIANT}.csv", index=False)
        pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(
            out_dir / f"test_{VARIANT}.csv", index=False)
    else:
        # A partial OOF cannot enter the stack, so it goes out under a name the
        # registry does not pick up.
        pd.DataFrame({ID: train[ID].iloc[scored_rows], "y": y[scored_rows],
                      "pred": oof[scored_rows]}).to_csv(
            out_dir / f"partial_oof_{VARIANT}.csv", index=False)

    (out_dir / f"metrics_{VARIANT}.json").write_text(json.dumps({
        "model": VARIANT,
        "architecture": CFG["arch"],
        "official_data_only": True,
        "public_predictions_used": False,
        "outer_folds": n_folds,
        "outer_folds_completed": folds_done,
        "complete": complete,
        "inner_encoding_folds": INNER_FOLDS,
        "net_seeds_averaged": NET_SEEDS,
        "epochs": args.epochs,
        "fold_records": records,
        "oof_auc": pooled,
        "runtime_seconds": time.time() - started,
        "submission_created": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    if Path("/kaggle").exists():
        sys.argv = ["experiment", "--threads", "4",
                    "--folds", str(CFG["folds"]), "--epochs", str(CFG["epochs"])]
    main()
