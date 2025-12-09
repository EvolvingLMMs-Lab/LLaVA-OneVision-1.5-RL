from dataclasses import dataclass, field

from areal.api.cli_args import DatasetConfig as BaseDatasetConfig
from areal.api.cli_args import GRPOConfig as BaseGRPOConfig
from areal.api.cli_args import PPOActorConfig as BasePPOActorConfig


@dataclass
class DatasetConfig(BaseDatasetConfig):
    split: str = field(
        default="train",
        metadata={"help": "Dataset split to use (e.g., 'train', 'valid', 'test')"},
    )
    ignore_prompt_type: bool = field(
        default=False,
        metadata={"help": "Whether to ignore prompt type"},
    )

@dataclass
class PPOActorConfig(BasePPOActorConfig):
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Level at which to compute importance sampling ratios. 'token': per-token ratios (standard PPO). 'sequence': sequence-level geometric mean of per-token ratios (GSPO).",
            "choices": ["token", "sequence"],
        },
    )

@dataclass
class GRPOConfig(BaseGRPOConfig):
    actor: PPOActorConfig = field(
        default_factory=PPOActorConfig,
    )
    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    valid_dataset: DatasetConfig | None = field(default=None)