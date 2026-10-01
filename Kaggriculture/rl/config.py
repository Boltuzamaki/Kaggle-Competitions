from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    seed: int = 7
    board_size: int = 10
    max_units: int = 12
    episode_steps: int = 720
    hidden_size: int = 256
    tile_channels: int = 24
    global_size: int = 96
    unit_feature_size: int = 20
    rollout_steps: int = 128
    actors: int = 6
    updates: int = 2000
    batch_sequences: int = 16
    epochs: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    snapshot_every: int = 50
    eval_every: int = 25
    checkpoint_dir: str = "rl_runs/checkpoints"
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)

