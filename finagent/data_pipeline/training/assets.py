from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DATASET_DIR = REPO_ROOT / "data" / "finagent"


@dataclass(frozen=True)
class TrainingAssets:
    train_config: Path
    deepspeed_config: Path
    dataset_info: Path
    dataset_jsonl: Path


class SFTTrainingPlan(BaseModel):
    model_name_or_path: str = "Qwen/Qwen2.5-14B-Instruct"
    dataset_name: str = "finagent_tool_calling_30k"
    train_config: str
    deepspeed_config: str
    dataset_info: str
    dataset_jsonl: str
    command: list[str] = Field(default_factory=list)
    ready_without_dataset: bool
    missing_assets: list[str] = Field(default_factory=list)


def default_training_assets() -> TrainingAssets:
    return TrainingAssets(
        train_config=CONFIG_DIR / "qwen2_5_14b_finagent_sft.yaml",
        deepspeed_config=CONFIG_DIR / "ds_z2_config.json",
        dataset_info=DATASET_DIR / "dataset_info.json",
        dataset_jsonl=DATASET_DIR / "finagent_tool_calling_30k.jsonl",
    )


def check_training_assets(
    assets: TrainingAssets | None = None,
    *,
    require_dataset: bool = False,
) -> list[str]:
    assets = assets or default_training_assets()
    required = [assets.train_config, assets.deepspeed_config, assets.dataset_info]
    if require_dataset:
        required.append(assets.dataset_jsonl)
    return [str(path) for path in required if not path.exists()]


def build_llamafactory_command(
    assets: TrainingAssets | None = None,
    *,
    executable: str = "llamafactory-cli",
) -> list[str]:
    assets = assets or default_training_assets()
    return [executable, "train", str(assets.train_config)]


def build_training_plan(
    assets: TrainingAssets | None = None,
    *,
    require_dataset: bool = False,
    executable: str = "llamafactory-cli",
) -> SFTTrainingPlan:
    assets = assets or default_training_assets()
    missing = check_training_assets(assets, require_dataset=require_dataset)
    return SFTTrainingPlan(
        train_config=str(assets.train_config),
        deepspeed_config=str(assets.deepspeed_config),
        dataset_info=str(assets.dataset_info),
        dataset_jsonl=str(assets.dataset_jsonl),
        command=build_llamafactory_command(assets, executable=executable),
        ready_without_dataset=not check_training_assets(assets, require_dataset=False),
        missing_assets=missing,
    )

