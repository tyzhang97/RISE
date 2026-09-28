"""Assemble parcel-level RISE outputs into model-ready Feather datasets."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .features import feature_basename
from .preprocessing import strip_subject_prefix

PathLike = Union[str, os.PathLike]
def _config_for_feature_root(
    config: Mapping[str, Any], feature_root: Optional[PathLike]
) -> Mapping[str, Any]:
    """Use an explicitly supplied processed-data root for this call."""
    if feature_root is None:
        return config
    selected = dict(config)
    selected["output_root"] = str(Path(feature_root))
    return selected


def _feature_stem(subject: str, run: str, config: Mapping[str, Any], mode: str) -> str:
    run_part = "mean" if run == "mean" else run
    return "sub-{}_{}{}".format(subject, run_part, feature_basename(config, mode))


def _feature_path(
    subject: str,
    run: str,
    config: Mapping[str, Any],
    mode: str,
    feature_name: str,
    extension: str = ".npy",
) -> Path:
    feature_dir = Path(config["output_root"]) / "indiv" / subject / "func" / run / "feature"
    return feature_dir / "{}_feature-{}{}".format(
        _feature_stem(subject, run, config, mode), feature_name, extension
    )


def _load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    values = np.load(str(path))
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError("Expected a 2D feature array, got {}: {}".format(values.shape, path))
    return np.asarray(values, dtype=float)


def _load_hctsa(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    values = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D hctsa table, got {}: {}".format(values.shape, path))
    return values


def _feature_blocks(
    subject: str, run: str, config: Mapping[str, Any]
) -> List[np.ndarray]:
    modes = [str(mode).upper() for mode in config.get("training_modes", ["NGR", "GSR"])]
    if modes != ["NGR", "GSR"]:
        raise ValueError(
            "training_modes must be ['NGR', 'GSR'] to match all_feature_names.csv"
        )

    blocks = []
    for mode in modes:
        blocks.append(_load_array(_feature_path(
            subject, run, config, mode,
            "{}_subcortical_fc".format(config["subcortical_atlas"]),
        )))
    for mode in modes:
        blocks.append(_load_array(_feature_path(
            subject, run, config, mode,
            "{}_cerebellar_fc".format(config["cerebellar_atlas"]),
        )))
    for mode in modes:
        blocks.append(_load_array(_feature_path(
            subject, run, config, mode, "aligned_gradients"
        )))
    for network_name in config["network_atlases"]:
        for mode in modes:
            blocks.append(_load_array(_feature_path(
                subject, run, config, mode, "{}_network_fc".format(network_name)
            )))
    for mode in modes:
        blocks.append(_load_hctsa(_feature_path(
            subject, run, config, mode, "HctsaTop49", extension=".csv"
        )))
    for mode in modes:
        blocks.append(np.concatenate([
            _load_array(_feature_path(subject, run, config, mode, metric))
            for metric in ("fALFF", "ALFF", "REHO")
        ], axis=1))
    return blocks


@lru_cache(maxsize=None)
def _cached_feature_names(path_string: str):
    path = Path(path_string)
    table = pd.read_csv(path)
    if "Feature_name" not in table.columns:
        raise ValueError("Feature_name column is missing from {}".format(path))
    names = table["Feature_name"].astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique: {}".format(path))
    return tuple(names)


@lru_cache(maxsize=None)
def _cached_parcel_metadata(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string)


def load_training_feature_names(config: Mapping[str, Any]) -> List[str]:
    """Load the canonical merged-feature column order from configuration."""
    return list(_cached_feature_names(str(config["feature_names_file"])))


def read_subject_training_frame(
    subject: Union[str, int],
    run: str,
    config: Mapping[str, Any],
    feature_root: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Read, normalize, and label all parcel features for one subject and run.

    Feature columns are z-scored across the subject's cortical parcels, matching
    the historical CorticalMetrics workflow. Non-finite values are imputed with
    the within-subject parcel mean before scaling; an all-missing feature raises.
    """
    subject_id = strip_subject_prefix(subject)
    subject_config = _config_for_feature_root(config, feature_root)
    arrays = _feature_blocks(subject_id, str(run), subject_config)
    row_counts = {array.shape[0] for array in arrays}
    if len(row_counts) != 1:
        raise ValueError(
            "Feature row counts differ for subject {} run {}: {}".format(
                subject_id, run, sorted(row_counts)
            )
        )
    values = np.concatenate(arrays, axis=1)
    feature_names = load_training_feature_names(config)
    if values.shape[1] != len(feature_names):
        raise ValueError(
            "Merged feature width {} does not match {} names in {}".format(
                values.shape[1], len(feature_names), config["feature_names_file"]
            )
        )

    values[~np.isfinite(values)] = np.nan
    means = np.nanmean(values, axis=0)
    missing_columns = np.flatnonzero(np.isnan(means))
    if missing_columns.size:
        missing_names = [feature_names[index] for index in missing_columns]
        raise ValueError(
            "Features contain no finite values for subject {} run {}: {}".format(
                subject_id, run, missing_names
            )
        )
    missing_rows, missing_cols = np.where(np.isnan(values))
    values[missing_rows, missing_cols] = means[missing_cols]

    standard_deviations = values.std(axis=0, ddof=0)
    nonconstant = standard_deviations > 0
    values[:, nonconstant] = (
        values[:, nonconstant] - means[nonconstant]
    ) / standard_deviations[nonconstant]
    values[:, ~nonconstant] = 0.0
    clip_value = float(config.get("training_clip_value", 5.0))
    values = np.clip(values, -clip_value, clip_value)

    metadata_path = Path(config["parcel_metadata_file"])
    metadata = _cached_parcel_metadata(str(metadata_path))
    if len(metadata) != values.shape[0]:
        raise ValueError(
            "Parcel metadata has {} rows but features have {}: {}".format(
                len(metadata), values.shape[0], metadata_path
            )
        )
    features = pd.DataFrame(values, columns=feature_names)
    subject_column = pd.Series([subject_id] * len(metadata), name="Subject_ID")
    return pd.concat(
        [subject_column, metadata.reset_index(drop=True), features], axis=1
    )


def read_training_subjects(path: PathLike) -> List[str]:
    """Read subject IDs from the first column of a headerless text/CSV file."""
    values = pd.read_csv(path, header=None, dtype=str).iloc[:, 0]
    return [strip_subject_prefix(value) for value in values if pd.notna(value)]


def find_missing_training_features(
    subjects: Iterable[Union[str, int]],
    run: str,
    config: Mapping[str, Any],
    feature_root: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Return missing feature files without loading feature arrays."""
    modes = [str(mode).upper() for mode in config.get("training_modes", ["NGR", "GSR"])]
    missing = []
    for subject in subjects:
        subject_id = strip_subject_prefix(subject)
        subject_config = _config_for_feature_root(config, feature_root)
        specifications = []
        for mode in modes:
            specifications.append((mode, "{}_subcortical_fc".format(config["subcortical_atlas"]), ".npy"))
        for mode in modes:
            specifications.append((mode, "{}_cerebellar_fc".format(config["cerebellar_atlas"]), ".npy"))
        for mode in modes:
            specifications.append((mode, "aligned_gradients", ".npy"))
        for network_name in config["network_atlases"]:
            for mode in modes:
                specifications.append((mode, "{}_network_fc".format(network_name), ".npy"))
        for mode in modes:
            specifications.append((mode, "HctsaTop49", ".csv"))
        for mode in modes:
            for metric in ("fALFF", "ALFF", "REHO"):
                specifications.append((mode, metric, ".npy"))
        for mode, feature_name, extension in specifications:
            path = _feature_path(
                subject_id, str(run), subject_config, mode, feature_name, extension=extension
            )
            if not path.is_file():
                missing.append({
                    "Subject_ID": subject_id,
                    "run": str(run),
                    "path": str(path),
                })
    return pd.DataFrame(missing, columns=["Subject_ID", "run", "path"])


def generate_training_feather(
    subjects: Iterable[Union[str, int]],
    run: str,
    split_name: str,
    config: Mapping[str, Any],
    output_path: Optional[PathLike] = None,
    show_progress: bool = True,
    feature_root: Optional[PathLike] = None,
) -> str:
    """Build one model-ready Feather file for a subject split and run."""
    subject_ids = [strip_subject_prefix(subject) for subject in subjects]
    if not subject_ids:
        raise ValueError("At least one subject is required")
    missing = find_missing_training_features(subject_ids, run, config, feature_root=feature_root)
    if not missing.empty:
        examples = "\n".join(missing["path"].head(10).tolist())
        raise FileNotFoundError(
            "Missing {} feature files for {} {}. First paths:\n{}".format(
                len(missing), split_name, run, examples
            )
        )
    frames = [
        read_subject_training_frame(subject, run, config, feature_root=feature_root)
        for subject in tqdm(
            subject_ids,
            desc="{} {}".format(split_name, run),
            unit="subject",
            disable=not show_progress,
        )
    ]
    dataset = pd.concat(frames, ignore_index=True)
    if dataset.isna().any().any():
        columns = dataset.columns[dataset.isna().any()].tolist()
        raise ValueError("Merged dataset contains NaN values: {}".format(columns))

    destination = Path(output_path) if output_path else (
        Path(config["training_output_root"])
        / str(run)
        / "{}_samples_parcel-level.feather".format(str(split_name).lower())
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_feather(str(destination))
    return str(destination)


def generate_group_mean_feather(
    sample_path: PathLike,
    config: Mapping[str, Any],
    output_path: Optional[PathLike] = None,
    groupby_column: str = "Parcel label",
    group_subject_id: str = "000000",
) -> str:
    """Average a sample-level Feather table across subjects for each parcel.

    Feature columns are averaged across subjects. Parcel metadata is retained
    from the first row for each parcel, and ``Subject_ID`` is replaced by the
    configured group identifier.
    """
    source = Path(sample_path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    data = pd.read_feather(source)
    feature_names = load_training_feature_names(config)
    required = ["Subject_ID", groupby_column] + feature_names
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Missing columns in {}: {}".format(source, missing_columns)
        )
    if data.duplicated(["Subject_ID", groupby_column]).any():
        raise ValueError(
            "Each subject must contain one row per {}: {}".format(
                groupby_column, source
            )
        )

    info_columns = [column for column in data.columns if column not in feature_names]
    aggregation = {column: "mean" for column in feature_names}
    aggregation.update({
        column: "first"
        for column in info_columns
        if column != groupby_column
    })
    group = data.groupby(groupby_column, as_index=False, sort=True).agg(aggregation)
    group = group[info_columns + feature_names]
    group["Subject_ID"] = str(group_subject_id)
    if group[feature_names].isna().any().any():
        columns = group[feature_names].columns[group[feature_names].isna().any()].tolist()
        raise ValueError("Group-mean features contain NaN values: {}".format(columns))

    if output_path is None:
        marker = "_samples_parcel-level.feather"
        if not source.name.endswith(marker):
            raise ValueError(
                "Cannot derive group output name from {}".format(source.name)
            )
        destination = source.with_name(
            source.name[:-len(marker)] + "_group-mean_parcel-level.feather"
        )
    else:
        destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    group.reset_index(drop=True).to_feather(str(destination))
    return str(destination)


def generate_group_mean_feathers(
    sample_outputs: Mapping[str, Mapping[str, PathLike]],
    config: Mapping[str, Any],
    groupby_column: str = "Parcel label",
    group_subject_id: str = "000000",
) -> Dict[str, Dict[str, str]]:
    """Generate group-mean Feather files for all run/split sample outputs."""
    outputs = {}
    for run, split_paths in sample_outputs.items():
        outputs[str(run)] = {}
        for split_name, sample_path in split_paths.items():
            outputs[str(run)][str(split_name)] = generate_group_mean_feather(
                sample_path,
                config,
                groupby_column=groupby_column,
                group_subject_id=group_subject_id,
            )
    return outputs


def generate_train_test_feathers(
    config: Mapping[str, Any],
    runs: Optional[Sequence[str]] = None,
    show_progress: bool = True,
) -> Dict[str, Dict[str, str]]:
    """Generate configured HCP train/test Feather files for every requested run."""
    split_files = {
        "train": config["train_subject_file"],
        "test": config["test_subject_file"],
    }
    selected_runs = list(runs if runs is not None else config["training_runs"])
    if not selected_runs:
        raise ValueError("At least one training run is required")
    outputs = {}
    for run in selected_runs:
        outputs[str(run)] = {}
        for split_name, subject_file in split_files.items():
            subjects = read_training_subjects(subject_file)
            outputs[str(run)][split_name] = generate_training_feather(
                subjects,
                str(run),
                split_name,
                config,
                show_progress=show_progress,
            )
    return outputs
