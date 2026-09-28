# Configuration and path handling
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .fetch import _fetch_atlas_repository_file, fetch_atlas

PathLike = Union[str, os.PathLike]
_PATH_KEYS = ("atlas_repo", "atlas_cache", "dataset_root", "output_root")


def _resolve_path(value: str, base_dir: Path) -> str:
    """Resolve a configured path relative to the configuration directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _strip_jsonc_comments(text: str) -> str:
    """Remove line comments from JSONC text while preserving quoted strings."""
    cleaned_lines = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        index = 0
        while index < len(line):
            character = line[index]
            if escaped:
                escaped = False
            elif character == "\\" and in_string:
                escaped = True
            elif character == '"':
                in_string = not in_string
            elif character == "/" and not in_string and line[index:index + 2] == "//":
                line = line[:index]
                break
            index += 1
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def load_config(
    config_path: PathLike,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and resolve an extraction configuration.

    Args:
        config_path: JSONC configuration file downloaded from the project or
            created by the user.
        overrides: Optional top-level values that replace entries loaded from JSON.

    Returns:
        A configuration dictionary in which path fields are absolute paths.
    """
    path = Path(config_path).expanduser().resolve()
    config = json.loads(_strip_jsonc_comments(path.read_text(encoding="utf-8")))

    if overrides:
        config.update(dict(overrides))

    for key in _PATH_KEYS:
        if config.get(key):
            config[key] = _resolve_path(config[key], path.parent)
    if config.get("subject_files"):
        config["subject_files"] = [
            _resolve_path(item, path.parent) for item in config["subject_files"]
        ]
    return config

# HCP subject and run discovery
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Union

import pandas as pd

PathLike = Union[str, os.PathLike]


def strip_subject_prefix(subject: Union[str, int]) -> str:
    """Return a subject identifier without an optional ``sub-`` prefix."""
    subject_id = str(subject)
    return subject_id[4:] if subject_id.startswith("sub-") else subject_id


def read_subject_ids(*files: PathLike) -> List[str]:
    """Read and combine subject identifiers.

    Args:
        *files: One-column text or CSV files containing subject identifiers.

    Returns:
        Subject identifiers in file order, preserving any duplicates.
    """
    subjects: List[str] = []
    for file in files:
        path = Path(file)
        if path.suffix.lower() == ".csv":
            table = pd.read_csv(path, dtype=str)
            if "subject_id" not in table.columns:
                raise ValueError(f"CSV subject file must contain a subject_id column: {path}")
            values = table["subject_id"]
        else:
            values = pd.read_csv(path, header=None, dtype=str).iloc[:, 0]
        subjects.extend(values.dropna().astype(str).tolist())
    return subjects


def _build_hcp_family_run_inputs(
    subjects: Iterable[Union[str, int]], config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Build extraction records for datasets using an HCP-style directory layout.

    Args:
        subjects: Subject identifiers, with or without a ``sub-`` prefix.
        config: Extraction configuration containing ``dataset_root``, ``runs``,
            ``tr``, and optionally ``bold_filename_template``. Each run must
            define ``output_name`` and either ``input_name`` or an ordered
            ``input_names`` list of alternative input stems.

    Returns:
        One dictionary per available standardized run. When alternatives are
        provided, the first existing input is selected and missing runs are omitted.
    """
    dataset_root = Path(config["dataset_root"])
    filename_template = config.get(
        "bold_filename_template", "{input_name}_hp2000_clean.nii.gz"
    )
    run_inputs: List[Dict[str, Any]] = []

    for subject in subjects:
        subject_id = strip_subject_prefix(subject)
        for run in config["runs"]:
            input_names = run.get("input_names", [run.get("input_name")])
            input_names = [name for name in input_names if name]
            for input_name in input_names:
                bold_file = (
                    dataset_root
                    / subject_id
                    / "MNINonLinear"
                    / "Results"
                    / input_name
                    / filename_template.format(input_name=input_name)
                )
                if bold_file.exists():
                    run_inputs.append(
                        {
                            "id": subject_id,
                            "nii_file": str(bold_file),
                            "input_name": input_name,
                            "output_name": run["output_name"],
                            "tr": float(config["tr"]),
                        }
                    )
                    break
    return run_inputs


def build_hcp_run_inputs(
    subjects: Iterable[Union[str, int]], config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Build run records for HCP Young Adult data.

    Args:
        subjects: Subject identifiers, with or without a ``sub-`` prefix.
        config: Resolved HCP extraction configuration.

    Returns:
        Run dictionaries accepted by :func:`rise.extract_dataset`.
    """
    return _build_hcp_family_run_inputs(subjects, config)


def build_hcpd_run_inputs(
    subjects: Iterable[Union[str, int]], config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """Build run records for HCP Development data.

    The HCP-D release contains a small number of subjects whose run stems use
    ``a`` or ``b`` suffixes. Ordered alternatives in the configuration map
    these files onto the same four standardized output run names.

    Args:
        subjects: HCP-D subject identifiers such as ``HCD0001305_V1_MR``.
        config: Resolved HCP-D extraction configuration.

    Returns:
        Run dictionaries accepted by :func:`rise.extract_dataset`.
    """
    return _build_hcp_family_run_inputs(subjects, config)


def build_fmriprep_run_inputs(
    subjects: Optional[Iterable[Union[str, int]]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build run records for a BIDS/fMRIPrep derivatives directory.

    Args:
        subjects: Optional subject identifiers with or without a ``sub-``
            prefix. When omitted, all ``sub-*`` directories under
            ``dataset_root`` are discovered automatically. For convenience,
            the configuration may be passed as the first positional argument.
        config: Configuration containing ``dataset_root``, ``fmriprep`` and
            ``runs``. Each run may provide ``session``, ``task``, ``run`` and
            ``space``; defaults are taken from the nested ``fmriprep`` mapping.

    Returns:
        One record per existing preprocessed BOLD image. Each record also
        contains the matching fMRIPrep confounds TSV path when available.
    """
    if config is None and isinstance(subjects, Mapping):
        config = subjects
        subjects = None
    if config is None:
        raise TypeError("config is required to discover fMRIPrep runs")

    dataset_root = Path(config["dataset_root"])
    if subjects is None:
        subjects = [
            path.name
            for path in sorted(dataset_root.glob("sub-*"))
            if path.is_dir()
        ]
    options = dict(config.get("fmriprep", {}))
    session = str(options.get("session", "ses-1"))
    task = str(options.get("task", "rest"))
    space = str(options.get("space", config.get("space", "MNI152NLin6Asym")))
    resolution = str(options.get("resolution", config.get("resolution", "2mm")))
    run_inputs: List[Dict[str, Any]] = []

    for subject in subjects:
        subject_id = strip_subject_prefix(subject)
        for run in config["runs"]:
            run_label = str(run["run"])
            output_name = str(run["output_name"])
            prefix = f"sub-{subject_id}_{session}_task-{task}_{run_label}"
            func_dir = dataset_root / f"sub-{subject_id}" / session / "func"
            bold_file = func_dir / (
                f"{prefix}_space-{space}_res-{resolution}_desc-preproc_bold.nii.gz"
            )
            confounds_file = func_dir / f"{prefix}_desc-confounds_timeseries.tsv"
            if bold_file.exists():
                record = {
                    "id": subject_id,
                    "nii_file": str(bold_file),
                    "output_name": output_name,
                    "tr": float(run.get("tr", config["tr"])),
                }
                if confounds_file.exists():
                    record["confounds_file"] = str(confounds_file)
                run_inputs.append(record)
    return run_inputs

# Atlas loading and preparation
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from nilearn import image
from scipy.spatial.distance import cdist

PathLike = Union[str, os.PathLike]


def find_annotation_file(atlas_dir: Path) -> Path:
    """Find an atlas annotation table while tolerating historical case changes."""
    for name in ("annot.csv", "Annot.csv"):
        candidate = atlas_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No annotation CSV found in {atlas_dir}")


def load_atlas_metadata(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Load metadata for all configured atlases.

    Args:
        config: Extraction configuration containing the atlas repository, spatial
            template, resolution, and atlas names.

    Returns:
        A dictionary keyed by atlas name. Each entry contains the image path,
        region labels, region names, and centroids when available.

    Raises:
        FileNotFoundError: If an atlas image or annotation table is missing.
        ValueError: If an annotation table lacks required label columns.
    """
    atlas_repo = config.get("atlas_repo")
    space = config["space"]
    resolution = config["resolution"]
    metadata: Dict[str, Dict[str, Any]] = {}

    for atlas_name in config["atlases"]:
        atlas_file, annotation_file = fetch_atlas(
            atlas_name,
            space=space,
            resolution=resolution,
            atlas_repo=atlas_repo,
        )
        annotation = pd.read_csv(annotation_file)
        required = {"Region label", "Region name"}
        missing = required.difference(annotation.columns)
        if missing:
            raise ValueError(f"{atlas_name} annotation is missing columns: {sorted(missing)}")

        atlas_info: Dict[str, Any] = {
            "atlas_path": str(atlas_file),
            "labels": annotation["Region label"].tolist(),
            "names": annotation["Region name"].tolist(),
        }
        centroid_columns = ["Centroid R", "Centroid A", "Centroid S"]
        if all(column in annotation.columns for column in centroid_columns):
            atlas_info["centroids"] = annotation[centroid_columns].to_numpy(float)
        metadata[atlas_name] = atlas_info
    return metadata


def validate_prepared_grid(reference_img, cache_dir: Path, atlases: Sequence[str]) -> None:
    """Validate cached images against a BOLD spatial grid.

    Args:
        reference_img: Niimg-like BOLD reference image.
        cache_dir: Directory containing ``Atlas_Merge`` and ``Atlas_Mask``.
        atlases: Atlas names expected in the cache.

    Returns:
        None.

    Raises:
        ValueError: If any cached shape or affine differs from the reference.
    """
    files = [cache_dir / "Atlas_Mask" / "Brain_mask.nii.gz"]
    files.extend(cache_dir / "Atlas_Merge" / name / "atlas.nii.gz" for name in atlases)
    for file in files:
        atlas_img = image.load_img(file)
        if atlas_img.shape[:3] != reference_img.shape[:3] or not np.allclose(
            atlas_img.affine, reference_img.affine
        ):
            raise ValueError(
                f"Prepared image grid does not match the BOLD reference: {file}. "
                "Run prepare_atlases(..., force=True)."
            )


def prepare_atlases(
    reference_file: PathLike,
    config: Mapping[str, Any],
    force: bool = False,
) -> Path:
    """Prepare atlases and a brain mask for extraction.

    Args:
        reference_file: Three- or four-dimensional NIfTI image defining the target
            shape and affine.
        config: Extraction configuration with atlas repository and cache settings.
        force: Resample and overwrite cached images when ``True``.

    Returns:
        Path to the atlas cache directory.

    Raises:
        FileNotFoundError: If a configured atlas or brain mask is unavailable.
        ValueError: If a prepared image does not match the reference grid.
    """
    cache_dir = Path(config["atlas_cache"])
    merge_dir = cache_dir / "Atlas_Merge"
    mask_dir = cache_dir / "Atlas_Mask"
    merge_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    reference_img = image.load_img(reference_file)
    if len(reference_img.shape) == 4:
        reference_img = image.mean_img(reference_img)

    for atlas_name, atlas_info in load_atlas_metadata(config).items():
        output_dir = merge_dir / atlas_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_atlas = output_dir / "atlas.nii.gz"
        output_labels = output_dir / "labels.csv"

        if force or not output_atlas.exists():
            atlas_img = image.resample_to_img(
                image.load_img(atlas_info["atlas_path"]),
                reference_img,
                interpolation="nearest",
            )
            atlas_img.to_filename(output_atlas)
        if force or not output_labels.exists():
            pd.DataFrame(
                {
                    "Region label": atlas_info["labels"],
                    "Region name": atlas_info["names"],
                }
            ).to_csv(output_labels, index=False)

    source_mask = _fetch_atlas_repository_file(
        config["brain_mask"],
        atlas_repo=config.get("atlas_repo"),
        description="Brain mask",
    )

    output_mask = mask_dir / "Brain_mask.nii.gz"
    if force or not output_mask.exists():
        mask_img = image.resample_to_img(
            image.load_img(source_mask), reference_img, interpolation="nearest"
        )
        mask_img.to_filename(output_mask)

    validate_prepared_grid(reference_img, cache_dir, config["atlases"])
    return cache_dir


def fill_zero_timeseries(
    timeseries: np.ndarray, labels: Sequence[Any], centroids: np.ndarray
) -> np.ndarray:
    """Replace empty parcel time series using nearby parcels.

    Args:
        timeseries: Array shaped ``(timepoints, parcels)``.
        labels: Parcel labels in the same order as the timeseries columns.
        centroids: Parcel coordinates shaped ``(parcels, 3)``.

    Returns:
        The input array with each all-zero column replaced by the mean of its two
        nearest non-empty parcels.

    Raises:
        ValueError: If two non-empty neighbors cannot be found or zero columns remain.
    """
    zero_columns = np.flatnonzero(np.all(timeseries == 0, axis=0))
    if zero_columns.size == 0:
        return timeseries

    distances = cdist(centroids, centroids)
    np.fill_diagonal(distances, np.inf)
    labels_array = np.asarray(labels)

    for column in zero_columns:
        # Search outward from the empty parcel until two usable parcels are found.
        neighbors = [
            index
            for index in np.argsort(distances[column])
            if np.sum(timeseries[:, index]) != 0
        ][:2]
        if len(neighbors) != 2:
            raise ValueError(f"Could not find two non-empty neighbors for column {column}")
        print(
            f"Filling empty parcel {labels_array[column]} from neighbors "
            f"{labels_array[neighbors].tolist()}"
        )
        timeseries[:, column] = np.mean(timeseries[:, neighbors], axis=1)

    if np.any(np.all(timeseries == 0, axis=0)):
        raise ValueError("All-zero parcel columns remain after nearest-neighbor filling")
    return timeseries


