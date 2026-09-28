"""Configuration loading for RISE neural-network scripts."""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Sequence

from ..fetch import (
    fetch_fine_grained_geodesic_distances,
    fetch_feature_names,
    fetch_multi_atlas_soft_labels,
)
from ._config_io import config_path_from_argv, load_jsonc_parameters


_PACKAGED_RESOURCE_KEYS = {
    "soft_labels",
    "geodesic_distances",
    "feature_names",
}

_PATH_KEYS = (
    "atlas_repo",
    "train_data",
    "test_data",
    "test_group_data",
    "models_root",
)


@dataclass
class TrainingConfig:
    train_data: Path
    test_data: Path
    soft_labels: Path
    geodesic_distances: Path
    feature_names: Path
    models_root: Path
    test_group_data: Optional[Path]
    model_name: str
    device: str
    repeats: int
    input_dim: int
    hidden_dims: Sequence[int]
    embedding_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_ratio: float
    num_workers: int
    scheduler_step_size: int
    scheduler_gamma: float
    early_stopping_patience: int
    geodesic_loss_weight: float
    embedding_l2_weight: float
    use_best_checkpoint: bool
    run_name: str

    @property
    def model_dir(self) -> Path:
        return self.models_root / self.model_name


def load_training_parameters(config_path: Path):
    path = Path(config_path)
    allowed = {
        field.name for field in fields(TrainingConfig)
    } - _PACKAGED_RESOURCE_KEYS
    # Fine-tuning presets can also be consumed by test.py and transform.py.
    allowed.add("pretrained_model_name")
    allowed.add("atlas_repo")
    values, resolved_path = load_jsonc_parameters(path, _PATH_KEYS, allowed)
    atlas_repo = values.pop("atlas_repo", None)
    values["soft_labels"] = fetch_multi_atlas_soft_labels(atlas_repo=atlas_repo)
    values["geodesic_distances"] = fetch_fine_grained_geodesic_distances(
        atlas_repo=atlas_repo
    )
    values["feature_names"] = fetch_feature_names()
    values.pop("pretrained_model_name", None)
    missing = sorted({field.name for field in fields(TrainingConfig)} - set(values))
    if missing:
        raise ValueError(
            "Missing neural-network configuration keys in {}: {}".format(
                resolved_path, ", ".join(missing)
            )
        )
    return values, resolved_path


def _add_boolean_switch(parser, name: str, default: bool, help_text: str) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--" + name, dest=destination, action="store_true", help=help_text)
    group.add_argument("--no-" + name, dest=destination, action="store_false", help="Disable: " + help_text)
    parser.set_defaults(**{destination: default})


def build_argument_parser(
    description: Optional[str] = None,
    argv: Optional[List[str]] = None,
) -> argparse.ArgumentParser:
    config_path = config_path_from_argv(argv)
    defaults, config_path = load_training_parameters(config_path)
    parser = argparse.ArgumentParser(
        description=description or "Configure a RISE neural-network step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSONC neural-network parameter file downloaded from the RISE repository.",
    )

    paths = parser.add_argument_group("input and output paths")
    paths.add_argument("--train-data", type=Path, default=defaults["train_data"])
    paths.add_argument("--test-data", type=Path, default=defaults["test_data"])
    paths.add_argument("--test-group-data", type=Path, default=defaults["test_group_data"])
    paths.add_argument("--models-root", type=Path, default=defaults["models_root"])
    parser.set_defaults(
        soft_labels=defaults["soft_labels"],
        geodesic_distances=defaults["geodesic_distances"],
        feature_names=defaults["feature_names"],
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default=defaults["device"], help="cuda, cuda:N, cpu, or auto")
    runtime.add_argument("--num-workers", type=int, default=defaults["num_workers"])
    runtime.add_argument("--run-name", default=defaults["run_name"])

    model = parser.add_argument_group("model")
    model.add_argument("--input-dim", type=int, default=defaults["input_dim"])
    model.add_argument("--model-name", default=defaults["model_name"])
    model.add_argument("--hidden-dims", type=int, nargs="+", default=defaults["hidden_dims"])
    model.add_argument("--embedding-dim", type=int, default=defaults["embedding_dim"])

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--repeats", type=int, default=defaults["repeats"])
    optimization.add_argument("--epochs", type=int, default=defaults["epochs"])
    optimization.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    optimization.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    optimization.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    optimization.add_argument("--validation-ratio", type=float, default=defaults["validation_ratio"])
    optimization.add_argument("--scheduler-step-size", type=int, default=defaults["scheduler_step_size"])
    optimization.add_argument("--scheduler-gamma", type=float, default=defaults["scheduler_gamma"])
    optimization.add_argument("--early-stopping-patience", type=int, default=defaults["early_stopping_patience"])
    optimization.add_argument("--geodesic-loss-weight", type=float, default=defaults["geodesic_loss_weight"])
    optimization.add_argument("--embedding-l2-weight", type=float, default=defaults["embedding_l2_weight"])
    _add_boolean_switch(optimization, "use-best-checkpoint", defaults["use_best_checkpoint"], "Use the checkpoint with the lowest validation loss.")
    return parser


def config_from_namespace(namespace: argparse.Namespace) -> TrainingConfig:
    values = vars(namespace).copy()
    values.pop("config", None)
    values.pop("split", None)
    return TrainingConfig(**values)


def validate_config(config: TrainingConfig) -> None:
    if not 0.0 < config.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1")
    if config.repeats < 1 or config.epochs < 1 or config.batch_size < 2:
        raise ValueError("repeats and epochs must be positive; batch-size must be at least 2")
    if not config.hidden_dims or any(width < 1 for width in config.hidden_dims):
        raise ValueError("--hidden-dims must contain positive integers")


def parse_training_config(
    argv: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> TrainingConfig:
    parser = build_argument_parser(description, argv=argv)
    config = config_from_namespace(parser.parse_args(argv))
    validate_config(config)
    return config
