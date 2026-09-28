"""Extract parcel embeddings from trained RISE checkpoints."""

from pathlib import Path

import pandas as pd

from .config import (
    TrainingConfig,
    build_argument_parser,
    config_from_namespace,
    validate_config,
)
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


def transform_dataset(
    config: TrainingConfig,
    data_path: Path,
    split_name: str,
    device,
) -> None:
    """Save left/right parcel embeddings for every trained repeat."""
    print("transforming {} data: {}".format(split_name, data_path))
    dataset, _ = load_paired_dataset(data_path, config)
    metadata = load_prediction_metadata(data_path)

    for repeat_dir in repeat_directories(config):
        model = load_model(config, checkpoint_path(config, repeat_dir), device)
        _, _, left_embedding, right_embedding = predict_dataset(
            model, dataset, config, device, pad_final_batch=True
        )
        results_dir = repeat_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        for hemi, embedding in (
            ("lh", left_embedding),
            ("rh", right_embedding),
        ):
            metadata_key = "Left" if hemi == "lh" else "Right"
            table = pd.concat(
                [
                    metadata[metadata_key].reset_index(drop=True),
                    pd.DataFrame(
                        embedding,
                        columns=[
                            "dim_{}".format(index + 1)
                            for index in range(embedding.shape[1])
                        ],
                    ),
                ],
                axis=1,
            )
            output = results_dir / "run_{}_{}_embeddings_{}.h5".format(
                config.run_name.lower(), split_name.lower(), hemi
            )
            table.to_hdf(output, key="data", mode="w")
        print("{} {} embeddings saved".format(repeat_dir.name, split_name))


def transform(config: TrainingConfig, split: str) -> None:
    """Extract train, test, or both sets of parcel embeddings."""

    config.model_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(config.model_dir, "transform.log")
    device = resolve_device(config.device)
    logger.info("device=%s split=%s", device, split)

    if split in ("train", "both"):
        transform_dataset(config, config.train_data, "train", device)
    if split in ("test", "both"):
        transform_dataset(config, config.test_data, "test", device)


def main(argv=None) -> None:
    parser = build_argument_parser(
        "Extract train and/or test parcel embeddings from RISE checkpoints.",
        argv=argv,
    )
    parser.add_argument(
        "--split",
        choices=("train", "test", "both"),
        default="both",
        help="Dataset whose embeddings should be exported.",
    )
    arguments = parser.parse_args(argv)
    split = arguments.split
    config = config_from_namespace(arguments)
    validate_config(config)
    transform(config, split)


if __name__ == "__main__":
    main()
