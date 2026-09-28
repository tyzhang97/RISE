"""Evaluate trained RISE checkpoints on group-mean parcel data."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import TrainingConfig, parse_training_config
from .utils import (
    checkpoint_path,
    configure_logger,
    load_model,
    load_paired_dataset,
    load_prediction_metadata,
    predict_dataset,
    repeat_directories,
    resolve_device,
)


def derived_group_path(test_data: Path) -> Path:
    """Derive the group-mean filename from a test-samples filename."""

    marker = "_samples_parcel-level"
    if marker not in test_data.name:
        raise ValueError(
            "Set --test-group-data because a group filename cannot be derived from {}".format(
                test_data
            )
        )
    return test_data.with_name(
        test_data.name.replace(marker, "_group-mean_parcel-level")
    )


def test_group(config: TrainingConfig) -> None:
    """Calculate per-atlas accuracy for every repeat on group-mean data.

    Individual test samples are not used here; they are reserved for embedding
    extraction in ``transform.py``.
    """
    config.model_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(config.model_dir, "test.log")
    device = resolve_device(config.device)
    group_path = config.test_group_data or derived_group_path(config.test_data)
    if not group_path.is_file():
        raise FileNotFoundError(
            "Group test data not found: {}. Run the feature-integration notebook first."
            .format(group_path)
        )

    logger.info("device=%s test_group_data=%s", device, group_path)
    dataset, _ = load_paired_dataset(group_path, config)
    metadata = load_prediction_metadata(group_path)
    left_rows = []
    right_rows = []

    for repeat_dir in repeat_directories(config):
        model = load_model(config, checkpoint_path(config, repeat_dir), device)
        left_pred, right_pred, _, _ = predict_dataset(model, dataset, config, device, pad_final_batch=True)
        left_true = dataset.left.hard_labels
        right_true = dataset.right.hard_labels
        results_dir = repeat_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        left_table = metadata["Left"].copy()
        right_table = metadata["Right"].copy()
        left_row = {"repeat": repeat_dir.name}
        right_row = {"repeat": repeat_dir.name}
        for index, atlas in enumerate(model.atlas_names):
            left_table["gt_" + atlas] = left_true[:, index]
            left_table["pred_" + atlas] = left_pred[:, index]
            right_table["gt_" + atlas] = right_true[:, index]
            right_table["pred_" + atlas] = right_pred[:, index]
            left_row[atlas + "_accuracy"] = float(
                (left_true[:, index] == left_pred[:, index]).mean()
            )
            right_row[atlas + "_accuracy"] = float(
                (right_true[:, index] == right_pred[:, index]).mean()
            )

        left_table.to_csv(
            results_dir / "test_group_pred_parcel_labels_lh.csv", index=False
        )
        right_table.to_csv(
            results_dir / "test_group_pred_parcel_labels_rh.csv", index=False
        )
        left_rows.append(left_row)
        right_rows.append(right_row)
        print(
            "{} mean accuracy: lh {:.4f}, rh {:.4f}".format(
                repeat_dir.name,
                np.mean(list(left_row.values())[1:]),
                np.mean(list(right_row.values())[1:]),
            )
        )

    pd.DataFrame(left_rows).to_csv(
        config.model_dir / "test_group_parcel_accuracy_repeat_list_lh.csv",
        index=False,
    )
    pd.DataFrame(right_rows).to_csv(
        config.model_dir / "test_group_parcel_accuracy_repeat_list_rh.csv",
        index=False,
    )


def main(argv=None) -> None:
    config = parse_training_config(
        argv,
        description="Test RISE checkpoints on group-mean parcel data.",
    )
    test_group(config)


if __name__ == "__main__":
    main()
