"""Shared data, DataLoader, checkpoint, and inference helpers."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
from .models import DEFAULT_ATLAS_CLASS_COUNTS, MultiAtlasSiameseNetwork




@dataclass
class HemisphereArrays:
    features: np.ndarray
    hard_labels: np.ndarray
    soft_label_lookup: np.ndarray
    parcel_indices: np.ndarray
    subject_ids: np.ndarray


class PairedHemisphereDataset(Dataset):
    """Return index-aligned left/right parcel samples.

    Soft labels are selected by parcel index in ``__getitem__``. This has the
    same effect as appending repeated soft labels to every subject row without
    storing those repeated values in memory.
    """

    def __init__(self, left: HemisphereArrays, right: HemisphereArrays) -> None:
        if len(left.features) != len(right.features):
            raise ValueError("Left and right hemisphere sample counts differ")
        if not np.array_equal(left.subject_ids, right.subject_ids):
            raise ValueError("Left and right samples are not aligned by subject")
        self.left = left
        self.right = right

    def __len__(self) -> int:
        return len(self.left.features)

    def __getitem__(self, index: int):
        left_parcel = self.left.parcel_indices[index]
        right_parcel = self.right.parcel_indices[index]
        return (
            self.left.features[index],
            self.left.hard_labels[index],
            self.left.soft_label_lookup[left_parcel],
            left_parcel,
            self.right.features[index],
            self.right.hard_labels[index],
            self.right.soft_label_lookup[right_parcel],
            right_parcel,
        )


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)


def load_feature_names(path: Path, input_dim: int) -> Sequence[str]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    table = pd.read_csv(path)
    if "Feature_name" not in table.columns:
        raise ValueError("Feature_name column is missing from {}".format(path))
    names = table["Feature_name"].astype(str).tolist()
    if input_dim > len(names):
        raise ValueError(
            "input_dim {} exceeds {} available features".format(input_dim, len(names))
        )
    return names[:input_dim]


def _soft_label_columns(table: pd.DataFrame, atlas_names: Sequence[str]) -> Sequence[str]:
    columns = []
    for atlas in atlas_names:
        prefix = atlas + "_label_"
        selected = [column for column in table.columns if str(column).startswith(prefix)]
        if not selected:
            raise ValueError("No soft-label columns found for atlas {}".format(atlas))
        selected.sort(key=lambda name: int(str(name).rsplit("_", 1)[1]))
        columns.extend(selected)
    return columns


def load_paired_dataset(
    data_path: Path,
    config: TrainingConfig,
) -> Tuple[PairedHemisphereDataset, Sequence[str]]:
    """Load and align features and labels for both hemispheres.

    Rows retain the source-table order, matching the historical implementation.
    One parcel-wise soft-label lookup is shared by every subject.
    """
    frame = read_table(data_path)
    feature_names = load_feature_names(config.feature_names, config.input_dim)
    atlas_names = tuple(DEFAULT_ATLAS_CLASS_COUNTS)
    required = ["Subject_ID", "Parcel label", "Parcel name", "Hemi"]
    required += list(feature_names)
    required += [name + " Train Label" for name in atlas_names]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns in {}: {}".format(data_path, missing))

    soft = pd.read_csv(config.soft_labels, index_col=0)
    if "Hemi" not in soft.columns:
        raise ValueError("Hemi column is missing from {}".format(config.soft_labels))
    soft.index = soft.index.astype(str)
    selected_soft_columns = _soft_label_columns(soft, atlas_names)
    expected_soft_width = sum(DEFAULT_ATLAS_CLASS_COUNTS.values())
    if len(selected_soft_columns) != expected_soft_width:
        raise ValueError(
            "Expected {} soft-label columns, found {}".format(
                expected_soft_width, len(selected_soft_columns)
            )
        )

    hemispheres = {}
    for hemi in ("Left", "Right"):
        subset = frame.loc[frame["Hemi"] == hemi].copy()
        subset["Subject_ID"] = subset["Subject_ID"].astype(str)
        subset = subset.reset_index(drop=True)
        counts = subset.groupby("Subject_ID", sort=False).size()
        if counts.empty or counts.nunique() != 1:
            raise ValueError(
                "Every subject must contain the same number of {} parcels".format(hemi)
            )

        parcel_metadata = subset.drop_duplicates("Parcel label").sort_values("Parcel label")
        hemi_soft = soft.loc[soft["Hemi"] == hemi, selected_soft_columns]
        parcel_soft_values = hemi_soft.reindex(
            parcel_metadata["Parcel name"].astype(str)
        ).to_numpy(dtype=np.float32)
        if np.isnan(parcel_soft_values).any():
            raise ValueError("Soft labels could not be matched to every {} parcel".format(hemi))

        label_columns = [name + " Train Label" for name in atlas_names]
        hard_labels = subset[label_columns].to_numpy(dtype=np.int64)
        for column, (atlas, class_count) in enumerate(DEFAULT_ATLAS_CLASS_COUNTS.items()):
            minimum = hard_labels[:, column].min()
            if minimum != 0:
                hard_labels[:, column] -= minimum
            if hard_labels[:, column].min() != 0 or hard_labels[:, column].max() >= class_count:
                raise ValueError(
                    "Hard labels for {} are outside [0, {})".format(atlas, class_count)
                )

        parcel_indices = subset["Parcel label"].to_numpy(dtype=np.int64) - 1
        soft_lookup = np.zeros(
            (int(frame["Parcel label"].max()), expected_soft_width), dtype=np.float32
        )
        lookup_indices = parcel_metadata["Parcel label"].to_numpy(dtype=np.int64) - 1
        soft_lookup[lookup_indices] = parcel_soft_values
        hemispheres[hemi] = HemisphereArrays(
            features=subset[list(feature_names)].to_numpy(dtype=np.float32),
            hard_labels=hard_labels,
            soft_label_lookup=soft_lookup,
            parcel_indices=parcel_indices,
            subject_ids=subset["Subject_ID"].to_numpy(),
        )

    return PairedHemisphereDataset(hemispheres["Left"], hemispheres["Right"]), feature_names


def load_prediction_metadata(data_path: Path) -> Dict[str, pd.DataFrame]:
    """Load parcel metadata in exactly the same order as the paired dataset."""
    frame = read_table(data_path)
    metadata = {}
    for hemi in ("Left", "Right"):
        subset = frame.loc[frame["Hemi"] == hemi].copy()
        subset["Subject_ID"] = subset["Subject_ID"].astype(str)
        subset = subset.reset_index(drop=True)
        metadata[hemi] = subset[["Subject_ID", "Parcel name", "Parcel label"]]
    return metadata


def subset_dataset(
    dataset: PairedHemisphereDataset,
    subject_ids: Sequence[str],
) -> PairedHemisphereDataset:
    selected = np.isin(dataset.left.subject_ids, np.asarray(subject_ids).astype(str))

    def take(values: HemisphereArrays) -> HemisphereArrays:
        return HemisphereArrays(
            values.features[selected],
            values.hard_labels[selected],
            values.soft_label_lookup,
            values.parcel_indices[selected],
            values.subject_ids[selected],
        )

    return PairedHemisphereDataset(take(dataset.left), take(dataset.right))


def make_loader(
    dataset: PairedHemisphereDataset,
    config: TrainingConfig,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Build a DataLoader using the shared batch and worker settings."""

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
        pin_memory=config.device.startswith("cuda"),
    )


def move_batch(batch, device: torch.device):
    return tuple(value.to(device) for value in batch)


def resolve_device(requested: str) -> torch.device:
    if requested.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Run this script on n08 or pass --device cpu."
        )
    return device


def configure_logger(output_dir: Path, filename: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rise.neural_network." + filename)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / filename, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def build_model(config: TrainingConfig, device: torch.device) -> MultiAtlasSiameseNetwork:
    return MultiAtlasSiameseNetwork(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        embedding_dim=config.embedding_dim,
    ).to(device)


def checkpoint_path(config: TrainingConfig, repeat_dir: Path) -> Path:
    if config.use_best_checkpoint:
        names = ("best_validloss_epoch.pth", "Best_validloss_epoch.pth")
    else:
        names = ("last_epoch.pth", "Last_epoch.pth")

    for name in names:
        path = repeat_dir / name
        if path.is_file():
            return path
    return repeat_dir / names[0]


def repeat_directories(config: TrainingConfig) -> Sequence[Path]:
    directories = {
        path
        for pattern in ("repeat_*", "Repeat_*")
        for path in config.model_dir.glob(pattern)
        if path.is_dir()
    }
    if not directories:
        raise FileNotFoundError(
            "No repeat directories were found under {}".format(config.model_dir)
        )
    return sorted(
        directories,
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )


def load_model(
    config: TrainingConfig,
    checkpoint: Path,
    device: torch.device,
) -> MultiAtlasSiameseNetwork:
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    model = build_model(config, device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_dataset(
    model: MultiAtlasSiameseNetwork,
    dataset: PairedHemisphereDataset,
    config: TrainingConfig,
    device: torch.device,
    pad_final_batch: bool = False,
):
    """Return predictions and embeddings for all left/right parcel pairs."""
    loader = make_loader(dataset, config, shuffle=False, drop_last=False)
    left_predictions = []
    right_predictions = []
    left_embeddings = []
    right_embeddings = []
    with torch.no_grad():
        for batch in loader:
            valid_batch_size = batch[0].shape[0]
            if pad_final_batch and valid_batch_size < config.batch_size:
                padding_size = config.batch_size - valid_batch_size
                batch = tuple(torch.cat((value, value[-1:].expand((padding_size,) + tuple(value.shape[1:]))), dim=0) for value in batch)
            left_x, _, _, _, right_x, _, _, _ = move_batch(batch, device)
            left_logits, left_embedding, right_logits, right_embedding = model(
                left_x, right_x
            )
            left_predictions.append(
                torch.stack(
                    [left_logits[name].argmax(dim=1) for name in model.atlas_names],
                    dim=1,
                )[:valid_batch_size].cpu().numpy()
            )
            right_predictions.append(
                torch.stack(
                    [right_logits[name].argmax(dim=1) for name in model.atlas_names],
                    dim=1,
                )[:valid_batch_size].cpu().numpy()
            )
            left_embeddings.append(left_embedding[:valid_batch_size].cpu().numpy())
            right_embeddings.append(right_embedding[:valid_batch_size].cpu().numpy())
    return (
        np.concatenate(left_predictions),
        np.concatenate(right_predictions),
        np.concatenate(left_embeddings),
        np.concatenate(right_embeddings),
    )
