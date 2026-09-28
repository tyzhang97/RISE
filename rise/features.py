import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from nilearn import image, surface
from scipy import ndimage
from tqdm.auto import tqdm

import joblib

from .fetch import (
    _fetch_feature_resource,
    fetch_atlas,
    fetch_fine_grained_surface,
    fetch_multi_atlas_label_mapping,
)
from .hctsa import load_hctsa_config
from .metrics import frequency_label
from .preprocessing import strip_subject_prefix

PathLike = Union[str, os.PathLike]
_FEATURE_PATH_KEYS = ("atlas_repo",)
_PACKAGED_FEATURE_RESOURCES = {
    "pfm_networks_priors_file": "PFM_networks_priors.csv",
    "ngr_gradient_reference_file": "group_NGR_reference_gradients.npy",
    "gsr_gradient_reference_file": "group_GSR_reference_gradients.npy",
}


@contextmanager
def _tqdm_joblib(progress):
    """Update a tqdm progress bar when joblib finishes a batch."""
    callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(callback):
        def __call__(self, *args, **kwargs):
            progress.update(n=self.batch_size)
            return callback.__call__(self, *args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress
    finally:
        joblib.parallel.BatchCompletionCallBack = callback
        progress.close()


def _run_single_subject(function, subject, args):
    """Run one subject without creating another process pool."""
    return function([subject], *args, n_jobs=1, show_progress=False)


def _parallel_subject_map(
    function, subjects, args, n_jobs, show_progress, description
):
    """Run subject-level jobs in parallel and combine their saved paths."""
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    workers = min(int(n_jobs), len(subjects))
    with _tqdm_joblib(tqdm(
        total=len(subjects), desc=description, unit="subject",
        disable=not show_progress,
    )):
        results = Parallel(n_jobs=workers)(
            delayed(_run_single_subject)(function, subject, args)
            for subject in subjects
        )
    saved = {}
    for result in results:
        saved.update(result)
    return saved


def _prepare_subject_jobs(subjects, function, args, n_jobs, show_progress, description):
    """Normalize subject IDs and start parallel jobs when requested."""
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be at least 1")
    subject_ids = [strip_subject_prefix(subject) for subject in subjects]
    if int(n_jobs) > 1 and len(subject_ids) > 1:
        return subject_ids, _parallel_subject_map(
            function, subject_ids, args, n_jobs, show_progress, description
        )
    return subject_ids, None


def _subject_iterator(subjects, show_progress, description):
    """Return subjects with an optional sequential progress bar."""
    if show_progress:
        return tqdm(subjects, desc=description, unit="subject")
    return subjects


def load_feature_config(
    config_path: PathLike,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load feature parameters and resolve file paths.

    Args:
        config_path: JSONC parameter file. 
        overrides: Optional top-level values applied after reading the file.

    Returns:
        Parameter dictionary with resolved paths.
    """
    path = Path(config_path).expanduser().resolve()
    config = load_hctsa_config(path, overrides=overrides)
    for key in _FEATURE_PATH_KEYS:
        if config.get(key):
            value = Path(config[key]).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            config[key] = str(value.resolve())
    config["atlas_mapping_file"] = str(
        fetch_multi_atlas_label_mapping(atlas_repo=config.get("atlas_repo"))
    )
    for key, filename in _PACKAGED_FEATURE_RESOURCES.items():
        config[key] = str(_fetch_feature_resource(filename))
    return config


def feature_timeseries_suffix(
    config: Mapping[str, Any], mode: str, atlas_name: Optional[str] = None
) -> str:
    """Return the exact RISE time-series suffix for an atlas and GSR mode.

    Args:
        config: Feature parameters.
        mode: Regression mode, either ``NGR`` or ``GSR``.
        atlas_name: Optional atlas override; defaults to ``cortical_atlas``.

    Returns:
        Filename suffix beginning with ``_space-`` and ending in
        ``_timeseries.npy``.
    """
    mode = mode.upper()
    if mode not in {"NGR", "GSR"}:
        raise ValueError("mode must be NGR or GSR, got {!r}".format(mode))
    atlas = atlas_name or config["cortical_atlas"]
    high = frequency_label(float(config["high_pass_hz"]))
    low = frequency_label(float(config["low_pass_hz"]))
    smooth = str(config["smoothing_fwhm_mm"]).zfill(2)
    detrend = "_detrend" if config["detrend"] else ""
    return (
        "_space-{space}_atlas-{atlas}_desc-filter{high}to{low}_sm{smooth}"
        "{detrend}_{mode}_timeseries.npy"
    ).format(
        space=config["space"], atlas=atlas, high=high, low=low,
        smooth=smooth, detrend=detrend, mode=mode
    )


def feature_basename(config: Mapping[str, Any], mode: str) -> str:
    """Return the feature filename segment following the run identifier."""
    suffix = feature_timeseries_suffix(config, mode)
    return suffix.split("space-{}".format(config["space"]), 1)[1].rsplit(
        "_timeseries.npy", 1
    )[0]


def discover_feature_runs(
    subject: Union[str, int], config: Mapping[str, Any], mode: str
) -> List[str]:
    """Discover runs containing the configured cortical atlas time series.

    Args:
        subject: Subject identifier with or without a ``sub-`` prefix.
        config: Feature parameters.
        mode: ``NGR`` or ``GSR``.

    Returns:
        Sorted standardized run names. The derived ``mean`` directory is excluded.
    """
    subject_id = strip_subject_prefix(subject)
    func_dir = Path(config["output_root"]) / "indiv" / subject_id / "func"
    suffix = feature_timeseries_suffix(config, mode)
    runs = []
    if not func_dir.is_dir():
        return runs
    for run_dir in sorted(path for path in func_dir.iterdir() if path.is_dir()):
        if run_dir.name == "mean":
            continue
        expected = run_dir / "data" / ("sub-{}_{}{}".format(subject_id, run_dir.name, suffix))
        if expected.exists():
            runs.append(run_dir.name)
    return runs


def _run_data_dir(subject: str, run: str, config: Mapping[str, Any]) -> Path:
    """Return a run's data directory under the lowercase RISE output layout."""
    return Path(config["output_root"]) / "indiv" / subject / "func" / run / "data"


def _run_feature_dir(subject: str, run: str, config: Mapping[str, Any]) -> Path:
    """Return a run's feature directory, creating it when needed."""
    path = Path(config["output_root"]) / "indiv" / subject / "func" / run / "feature"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timeseries_path(
    subject: str, run: str, atlas: str, mode: str, config: Mapping[str, Any]
) -> Path:
    """Return one atlas time-series path for a subject, run, and mode."""
    suffix = feature_timeseries_suffix(config, mode, atlas_name=atlas)
    return _run_data_dir(subject, run, config) / "sub-{}_{}{}".format(subject, run, suffix)


def _save_mean_array(
    arrays: Sequence[np.ndarray], output_path: PathLike
) -> np.ndarray:
    """Average run arrays, save the result, and return it.

    Args:
        arrays: Run-level arrays with identical shapes.
        output_path: Destination ``.npy`` path.

    Returns:
        Arithmetic mean across runs.
    """
    if not arrays:
        raise ValueError("Cannot compute a run mean from an empty array list")
    mean_array = np.mean(np.asarray(arrays), axis=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output), mean_array)
    return mean_array


def compute_correlation_matrix(timeseries: np.ndarray) -> np.ndarray:
    """Compute the FC matrix from regional time series."""
    if timeseries.ndim != 2:
        raise ValueError("Expected a 2D time-series matrix, got {}".format(timeseries.shape))
    matrix = np.corrcoef(timeseries, rowvar=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix.astype(np.float32)


def extract_cortical_fc(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    mode: str,
    n_jobs: int = 32,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """Compute cortical FC for each run and average it when multiple runs exist.

    Args:
        subjects: Subject identifiers.
        config: Feature parameters.
        mode: ``NGR`` or ``GSR``.
        n_jobs: Number of subjects processed in parallel.
        show_progress: Whether to display subject-level progress.

    Returns:
        Dictionary keyed by subject ID. Each value lists the saved FC paths.
        A mean path is included only when the subject has multiple runs.
    """
    subjects, parallel_result = _prepare_subject_jobs(
        subjects, extract_cortical_fc, (config, mode), n_jobs,
        show_progress, "Cortical FC ({})".format(mode.upper())
    )
    if parallel_result is not None:
        return parallel_result
    atlas = config["cortical_atlas"]
    basename = feature_basename(config, mode)
    skip_existing = bool(config.get("skip_existing", True))
    saved = {}
    for subject in _subject_iterator(subjects, show_progress, "{} ({})".format("extract_cortical_fc", mode.upper())):
        subject_id = strip_subject_prefix(subject)
        run_matrices = []
        paths = []
        for run in discover_feature_runs(subject_id, config, mode):
            output = _run_data_dir(subject_id, run, config) / (
                "sub-{}_{}{}_fc.npy".format(subject_id, run, basename)
            )
            if skip_existing and output.exists():
                matrix = np.load(str(output))
            else:
                timeseries = np.load(
                    str(_timeseries_path(subject_id, run, atlas, mode, config))
                )
                matrix = compute_correlation_matrix(timeseries)
                np.save(str(output), matrix)
            run_matrices.append(matrix)
            paths.append(str(output))
        if len(run_matrices) > 1:
            mean_output = _run_data_dir(subject_id, "mean", config) / (
                "sub-{}_mean{}_fc.npy".format(subject_id, basename)
            )
            _save_mean_array(run_matrices, mean_output)
            paths.append(str(mean_output))
        saved[subject_id] = paths
    return saved


def extract_functional_gradients(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    mode: str,
    n_jobs: int = 32,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """Compute individual gradients and align them to the HCP reference.

    Args:
        subjects: Subject identifiers.
        config: Feature parameters and NGR/GSR reference paths.
        mode: ``NGR`` or ``GSR``.
        n_jobs: Number of subjects processed in parallel.
        show_progress: Whether to display subject-level progress.

    Returns:
        Dictionary keyed by subject ID. Each value lists the saved aligned
        gradient paths. A mean-run path is included only for multiple runs.
    """
    subjects, parallel_result = _prepare_subject_jobs(
        subjects, extract_functional_gradients, (config, mode), n_jobs,
        show_progress, "Functional gradients ({})".format(mode.upper())
    )
    if parallel_result is not None:
        return parallel_result
    from brainspace.gradient import GradientMaps, alignment

    mode = mode.upper()
    basename = feature_basename(config, mode)
    reference_key = "{}_gradient_reference_file".format(mode.lower())
    reference = np.load(str(config[reference_key]))
    retained = int(config.get("gradient_components", 20))
    embedding = int(config.get("gradient_embedding_components", 30))
    if reference.shape != (7340, embedding):
        raise ValueError(
            "Expected reference shape (7340, {}), got {}".format(
                embedding, reference.shape
            )
        )
    saved = {}
    for subject in _subject_iterator(subjects, show_progress, "{} ({})".format("extract_functional_gradients", mode.upper())):
        subject_id = strip_subject_prefix(subject)
        paths = []
        runs = discover_feature_runs(subject_id, config, mode)
        if len(runs) > 1:
            runs.append("mean")
        for run in runs:
            fc_path = _run_data_dir(subject_id, run, config) / (
                "sub-{}_{}{}_fc.npy".format(subject_id, run, basename)
            )
            if not fc_path.exists():
                continue
            gradient_map = GradientMaps(
                n_components=embedding, approach="dm", kernel="normalized_angle",
                random_state=int(config.get("gradient_random_state", 0)),
            )
            gradient_map.fit(
                np.load(str(fc_path)),
                sparsity=float(config.get("gradient_sparsity", 0.9)),
                n_iter=int(config.get("gradient_n_iter", 10)),
            )
            raw = gradient_map.gradients_
            aligned = alignment.procrustes_alignment([raw], reference)[0][:, :retained]
            folder = _run_feature_dir(subject_id, run, config)
            raw_path = folder / "sub-{}_{}{}_feature-raw_gradients.npy".format(
                subject_id, run, basename
            )
            aligned_path = folder / "sub-{}_{}{}_feature-aligned_gradients.npy".format(
                subject_id, run, basename
            )
            np.save(str(raw_path), raw[:, :retained])
            np.save(str(aligned_path), aligned)
            paths.append(str(aligned_path))
        saved[subject_id] = paths
    return saved


def load_network_priors(network_name: str, config: Mapping[str, Any]) -> np.ndarray:
    """Load parcel-to-network priors for Yeo7, Yeo17, or PFM_networks.

    Args:
        network_name: One of the configured network names.
        config: Atlas mapping and PFM network paths.

    Returns:
        Array shaped cortical parcels by networks.
    """
    if network_name == "PFM_networks":
        return pd.read_csv(config["pfm_networks_priors_file"]).to_numpy(dtype=float)
    if network_name not in {"Yeo7", "Yeo17"}:
        raise ValueError("Unsupported cortical network atlas: {}".format(network_name))
    mapping = pd.read_csv(config["atlas_mapping_file"])
    count = 7 if network_name == "Yeo7" else 17
    labels = mapping["{} Train Label".format(network_name)].to_numpy(dtype=int)
    priors = np.zeros((len(labels), count), dtype=float)
    priors[np.arange(len(labels)), np.mod(labels, count)] = 1.0
    return priors


def _cross_correlation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return Pearson correlations between columns of two time-series matrices."""
    combined = np.hstack((first, second))
    correlation = np.corrcoef(combined, rowvar=False)
    result = correlation[: first.shape[1], first.shape[1] :]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extract_cortico_network_fc(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    mode: str,
    network_name: str,
    n_jobs: int = 32,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """Compute cortical-parcel connectivity to large-scale cortical networks.

    Args:
        subjects: Subject identifiers.
        config: Feature parameters.
        mode: ``NGR`` or ``GSR``.
        n_jobs: Number of subjects processed in parallel.
        show_progress: Whether to display subject-level progress.
        network_name: ``Yeo7``, ``Yeo17``, or ``PFM_networks``.

    Returns:
        Dictionary keyed by subject ID. Each value lists the saved cortico-network
        FC paths. A mean path is included only for multiple runs.
    """
    subjects, parallel_result = _prepare_subject_jobs(
        subjects, extract_cortico_network_fc, (config, mode, network_name),
        n_jobs, show_progress,
        "{} FC ({})".format(network_name, mode.upper())
    )
    if parallel_result is not None:
        return parallel_result
    priors = load_network_priors(network_name, config)
    denominator = priors.sum(axis=0, keepdims=True)
    if np.any(denominator == 0):
        raise ValueError("Network priors contain an empty column: {}".format(network_name))
    atlas = config["cortical_atlas"]
    basename = feature_basename(config, mode)
    saved = {}
    for subject in _subject_iterator(subjects, show_progress, "{} ({})".format("extract_cortico_network_fc", mode.upper())):
        subject_id = strip_subject_prefix(subject)
        arrays = []
        paths = []
        for run in discover_feature_runs(subject_id, config, mode):
            cortical = np.load(str(_timeseries_path(subject_id, run, atlas, mode, config)))
            if cortical.shape[1] != priors.shape[0]:
                raise ValueError("Prior rows do not match cortical parcels")
            network_ts = np.matmul(cortical, priors) / denominator
            feature = _cross_correlation(cortical, network_ts)
            output = _run_feature_dir(subject_id, run, config) / (
                "sub-{}_{}{}_feature-{}_network_fc.npy".format(
                    subject_id, run, basename, network_name
                )
            )
            np.save(str(output), feature)
            arrays.append(feature)
            paths.append(str(output))
        if len(arrays) > 1:
            mean_path = _run_feature_dir(subject_id, "mean", config) / (
                "sub-{}_mean{}_feature-{}_network_fc.npy".format(
                    subject_id, basename, network_name
                )
            )
            _save_mean_array(arrays, mean_path)
            paths.append(str(mean_path))
        saved[subject_id] = paths
    return saved


def extract_subcortical_cerebellar_fc(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    mode: str,
    target_atlas: str,
    feature_name: str,
    n_jobs: int = 32,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """Compute subcortical or cerebellar connectivity with cortical parcels.

    Args:
        subjects: Subject identifiers.
        config: Feature parameters.
        mode: ``NGR`` or ``GSR``.
        n_jobs: Number of subjects processed in parallel.
        show_progress: Whether to display subject-level progress.
        target_atlas: ``Tianye`` or ``Buckner``.
        feature_name: Output label, such as ``subcortical_fc``.

    Returns:
        Dictionary keyed by subject ID. Each value lists the saved subcortical or
        cerebellar FC paths. A mean path is included only for multiple runs.
    """
    subjects, parallel_result = _prepare_subject_jobs(
        subjects, extract_subcortical_cerebellar_fc,
        (config, mode, target_atlas, feature_name), n_jobs, show_progress,
        "{} ({})".format(feature_name, mode.upper())
    )
    if parallel_result is not None:
        return parallel_result
    cortical_atlas = config["cortical_atlas"]
    basename = feature_basename(config, mode)
    saved = {}
    for subject in _subject_iterator(subjects, show_progress, "{} ({})".format("extract_subcortical_cerebellar_fc", mode.upper())):
        subject_id = strip_subject_prefix(subject)
        arrays = []
        paths = []
        for run in discover_feature_runs(subject_id, config, mode):
            cortical = np.load(str(_timeseries_path(subject_id, run, cortical_atlas, mode, config)))
            target_path = _timeseries_path(subject_id, run, target_atlas, mode, config)
            if not target_path.exists():
                raise FileNotFoundError(str(target_path))
            target = np.load(str(target_path))
            feature = _cross_correlation(cortical, target)
            output = _run_feature_dir(subject_id, run, config) / (
                "sub-{}_{}{}_feature-{}.npy".format(subject_id, run, basename, feature_name)
            )
            np.save(str(output), feature)
            arrays.append(feature)
            paths.append(str(output))
        if len(arrays) > 1:
            mean_path = _run_feature_dir(subject_id, "mean", config) / (
                "sub-{}_mean{}_feature-{}.npy".format(subject_id, basename, feature_name)
            )
            _save_mean_array(arrays, mean_path)
            paths.append(str(mean_path))
        saved[subject_id] = paths
    return saved


def extract_roi_means(
    atlas_path: PathLike, metric_path: PathLike, labels: Sequence[int]
) -> np.ndarray:
    """Average a voxelwise metric within each requested atlas parcel.

    Args:
        atlas_path: Label-image path.
        metric_path: Voxelwise ReHo, ALFF, or fALFF image.
        labels: Region labels in the desired output order.

    Returns:
        Array shaped regions by one. Images are resampled with nearest-neighbor
        interpolation only when the atlas grid differs from the metric grid.
    """
    atlas_img = image.load_img(str(atlas_path))
    metric_img = image.load_img(str(metric_path))
    if atlas_img.shape != metric_img.shape or not np.allclose(atlas_img.affine, metric_img.affine):
        atlas_img = image.resample_to_img(atlas_img, metric_img, interpolation="nearest")
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    metric_data = metric_img.get_fdata()
    values = ndimage.mean(
        metric_data,
        labels=atlas_data,
        index=np.asarray(labels, dtype=np.int32),
    )
    return np.asarray(values, dtype=float)[:, np.newaxis]


def extract_classical_metrics(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    mode: str,
    n_jobs: int = 32,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """Parcel ReHo, ALFF, and fALFF normalized voxel maps.

    Args:
        subjects: Subject identifiers.
        config: Feature parameters and atlas paths.
        mode: ``NGR`` or ``GSR``.
        n_jobs: Number of subjects processed in parallel.
        show_progress: Whether to display subject-level progress.

    Returns:
        Dictionary keyed by subject ID. Each value lists the saved ReHo, ALFF,
        and fALFF paths. Mean paths are included only for multiple runs.
    """
    subjects, parallel_result = _prepare_subject_jobs(
        subjects, extract_classical_metrics, (config, mode), n_jobs,
        show_progress, "Classical metrics ({})".format(mode.upper())
    )
    if parallel_result is not None:
        return parallel_result
    atlas = config["cortical_atlas"]
    atlas_path, annotation_path = fetch_atlas(
        atlas,
        space=config["space"],
        resolution=config["resolution"],
        atlas_repo=config.get("atlas_repo"),
    )
    atlas_info = pd.read_csv(annotation_path)
    labels = atlas_info["Region label"].astype(int).tolist()
    basename = feature_basename(config, mode)
    high = frequency_label(float(config["high_pass_hz"]))
    low = frequency_label(float(config["low_pass_hz"]))
    smooth = str(config["smoothing_fwhm_mm"]).zfill(2)
    detrend = "_detrend" if config["detrend"] else ""
    volume_stem = "_space-{}_desc-filter{}to{}_sm{}{}_{}".format(
        config["space"], high, low, smooth, detrend, mode.upper()
    )
    saved = {}
    for subject in _subject_iterator(subjects, show_progress, "{} ({})".format("extract_classical_metrics", mode.upper())):
        subject_id = strip_subject_prefix(subject)
        metric_runs = {"REHO": [], "ALFF": [], "fALFF": []}
        paths = []
        for run in discover_feature_runs(subject_id, config, mode):
            data_dir = _run_data_dir(subject_id, run, config)
            feature_dir = _run_feature_dir(subject_id, run, config)
            for metric in ("REHO", "ALFF", "fALFF"):
                metric_path = data_dir / "sub-{}_{}{}_{}_norm.nii.gz".format(
                    subject_id, run, volume_stem, metric
                )
                if not metric_path.exists():
                    raise FileNotFoundError(str(metric_path))
                parcel_values = extract_roi_means(atlas_path, metric_path, labels)
                output = feature_dir / "sub-{}_{}{}_feature-{}.npy".format(
                    subject_id, run, basename, metric
                )
                np.save(str(output), parcel_values)
                metric_runs[metric].append(parcel_values)
                paths.append(str(output))
        for metric, arrays in metric_runs.items():
            if len(arrays) > 1:
                mean_path = _run_feature_dir(subject_id, "mean", config) / (
                    "sub-{}_mean{}_feature-{}.npy".format(subject_id, basename, metric)
                )
                _save_mean_array(arrays, mean_path)
                paths.append(str(mean_path))
        saved[subject_id] = paths
    return saved


def hctsa_operation_names(ops_path: PathLike) -> List[str]:
    """Read output names from the configured hctsa operation-list file."""
    names = []
    for line in Path(ops_path).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            names.append(fields[1])
    if not names:
        raise ValueError("No hctsa operations found in {}".format(ops_path))
    return names


def read_hctsa_output(
    mat_path: PathLike, operation_names: Sequence[str], region_names: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read hctsa v7.3 output into labeled data, quality, and timing tables.

    Args:
        mat_path: MATLAB output containing ``TS_DataMat``, ``TS_Quality``, and
            ``TS_CalcTime``.
        operation_names: Feature names in hctsa output-column order.
        region_names: Parcel names in time-series row order.

    Returns:
        Three region-by-feature DataFrames for values, quality labels, and time.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "Reading hctsa v7.3 MAT outputs requires h5py. "
            "Install h5py in the environment used to summarize hctsa features."
        )

    with h5py.File(str(mat_path), "r") as handle:
        data = np.asarray(handle["TS_DataMat"]).T
        quality = np.asarray(handle["TS_Quality"]).T
        timing = np.asarray(handle["TS_CalcTime"]).T
    if data.shape != (len(region_names), len(operation_names)):
        raise ValueError(
            "Unexpected hctsa shape {} for {} regions and {} operations".format(
                data.shape, len(region_names), len(operation_names)
            )
        )
    kwargs = {"index": list(region_names), "columns": list(operation_names)}
    return (
        pd.DataFrame(data, **kwargs),
        pd.DataFrame(quality, **kwargs),
        pd.DataFrame(timing, **kwargs),
    )


def summarize_hctsa_features(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    mode: str,
    n_jobs: int = 32,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """Convert run-level hctsa MAT outputs to labeled CSV features and means.

    Args:
        subjects: Subject identifiers.
        config: Feature parameters and hctsa operation paths.
        mode: ``NGR`` or ``GSR``.
        n_jobs: Number of subjects processed in parallel.
        show_progress: Whether to display subject-level progress.

    Returns:
        Dictionary keyed by subject ID. Each value lists the saved hctsa CSV
        paths. A mean path is included only for multiple runs.
    """
    subjects, parallel_result = _prepare_subject_jobs(
        subjects, summarize_hctsa_features, (config, mode), n_jobs,
        show_progress, "hctsa features ({})".format(mode.upper())
    )
    if parallel_result is not None:
        return parallel_result
    atlas = config["cortical_atlas"]
    _, annotation_path = fetch_atlas(
        atlas,
        space=config["space"],
        resolution=config["resolution"],
        atlas_repo=config.get("atlas_repo"),
    )
    info = pd.read_csv(annotation_path)
    region_names = info["Region name"].astype(str).tolist()
    operations = hctsa_operation_names(config["hctsa_ops_file"])
    basename = feature_basename(config, mode)
    saved = {}
    for subject in _subject_iterator(subjects, show_progress, "{} ({})".format("summarize_hctsa_features", mode.upper())):
        subject_id = strip_subject_prefix(subject)
        run_values = []
        paths = []
        for run in discover_feature_runs(subject_id, config, mode):
            stem = "sub-{}_{}{}".format(
                subject_id, run, feature_timeseries_suffix(config, mode).rsplit(".npy", 1)[0]
            )
            mat_path = _run_data_dir(subject_id, run, config) / "hctsa" / "outFeatures" / "all" / (
                "{}_out.mat".format(stem)
            )
            if not mat_path.exists():
                raise FileNotFoundError(str(mat_path))
            data, _, _ = read_hctsa_output(mat_path, operations, region_names)
            data.columns = ["Hctsa_{}".format(name) for name in data.columns]
            data.index.name = "Region name"
            output = _run_feature_dir(subject_id, run, config) / (
                "sub-{}_{}{}_feature-HctsaTop49.csv".format(subject_id, run, basename)
            )
            data.to_csv(str(output))
            run_values.append(data)
            paths.append(str(output))
        if len(run_values) > 1:
            mean_values = np.mean(np.asarray([frame.values for frame in run_values]), axis=0)
            mean_frame = pd.DataFrame(
                mean_values, index=run_values[0].index, columns=run_values[0].columns
            )
            mean_frame.index.name = "Region name"
            mean_path = _run_feature_dir(subject_id, "mean", config) / (
                "sub-{}_mean{}_feature-HctsaTop49.csv".format(subject_id, basename)
            )
            mean_frame.to_csv(str(mean_path))
            paths.append(str(mean_path))
        saved[subject_id] = paths
    return saved



def extract_all_features(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    modes: Optional[Sequence[str]] = None,
    n_jobs: int = 32,
    show_progress: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Extract all RISE feature families for each regression mode.

    Args:
        subjects: Subject identifiers.
        config: Feature parameters.
        modes: Regression modes to process. Defaults to ``gsr_modes`` from the
            parameter file.
        n_jobs: Number of subjects processed in parallel by each feature step.
        show_progress: Whether to display category and subject-level progress.

    Returns:
        Dictionary keyed by regression mode and feature category. Each category
        contains the saved output paths returned by its extraction function.
    """
    subject_ids = [strip_subject_prefix(subject) for subject in subjects]
    if not subject_ids:
        raise ValueError("At least one subject is required")
    selected_modes = [
        str(mode).upper()
        for mode in (modes if modes is not None else config.get("gsr_modes", ["NGR", "GSR"]))
    ]
    invalid_modes = [mode for mode in selected_modes if mode not in {"NGR", "GSR"}]
    if invalid_modes:
        raise ValueError("modes must contain only NGR or GSR: {}".format(invalid_modes))

    network_names = list(config["network_atlases"])
    steps_per_mode = 6 + len(network_names)
    total_steps = len(selected_modes) * steps_per_mode
    completed_steps = [0]

    def report_completed(mode, feature_label):
        completed_steps[0] += 1
        if show_progress:
            tqdm.write(
                "[{}/{}] Completed {} ({})".format(
                    completed_steps[0], total_steps, feature_label, mode
                )
            )

    if show_progress:
        tqdm.write(
            "Extracting all features for {} subjects and {} modes ({} steps).".format(
                len(subject_ids), len(selected_modes), total_steps
            )
        )

    outputs = {}
    for mode in selected_modes:
        cortical_fc = extract_cortical_fc(
            subject_ids, config, mode, n_jobs=n_jobs, show_progress=show_progress
        )
        report_completed(mode, "cortical FC")
        gradients = extract_functional_gradients(
            subject_ids, config, mode, n_jobs=n_jobs, show_progress=show_progress
        )
        report_completed(mode, "functional gradients")
        network_fc = {}
        for network_name in network_names:
            network_fc[network_name] = extract_cortico_network_fc(
                subject_ids, config, mode, network_name,
                n_jobs=n_jobs, show_progress=show_progress,
            )
            report_completed(mode, "{} network FC".format(network_name))
        subcortical_fc = extract_subcortical_cerebellar_fc(
            subject_ids, config, mode, config["subcortical_atlas"],
            "{}_subcortical_fc".format(config["subcortical_atlas"]),
            n_jobs=n_jobs, show_progress=show_progress,
        )
        report_completed(mode, "cortico-subcortical FC")
        cerebellar_fc = extract_subcortical_cerebellar_fc(
            subject_ids, config, mode, config["cerebellar_atlas"],
            "{}_cerebellar_fc".format(config["cerebellar_atlas"]),
            n_jobs=n_jobs, show_progress=show_progress,
        )
        report_completed(mode, "cortico-cerebellar FC")
        hctsa = summarize_hctsa_features(
            subject_ids, config, mode, n_jobs=n_jobs, show_progress=show_progress
        )
        report_completed(mode, "hctsa temporal dynamics")
        classical = extract_classical_metrics(
            subject_ids, config, mode, n_jobs=n_jobs, show_progress=show_progress
        )
        report_completed(mode, "classical local metrics")
        outputs[mode] = {
            "cortical_fc": cortical_fc,
            "functional_gradients": gradients,
            "cortico_network_fc": network_fc,
            "subcortical_fc": subcortical_fc,
            "cerebellar_fc": cerebellar_fc,
            "hctsa": hctsa,
            "classical_metrics": classical,
        }
    if show_progress:
        tqdm.write("All feature extraction steps completed.")
    return outputs


def visualize_surface_map(
    values: np.ndarray,
    config: Mapping[str, Any],
    title: str,
    output_file: Optional[PathLike] = None,
    cmap: str = "RdBu_r",
    max_value: Optional[float] = None,
):
    """Render one parcel-wise feature as a whole-cortex surface map.

    Args:
        values: One scalar per cortical parcel in atlas row order. Slice a
            two-dimensional feature matrix to one column before calling.
        config: Feature parameters.
        title: Figure title.
        output_file: Optional PNG destination.
        cmap: Matplotlib colormap name.
        max_value: Optional absolute color limit. The maximum absolute value of
            ``values`` is used when omitted.

    Returns:
        Matplotlib figure containing the rendered cortical surface map.
    """
    from .vis.surf import visualize_surface_32k_fs_LR

    vector = np.asarray(values).squeeze()
    atlas = config["cortical_atlas"]
    _, annotation_path = fetch_atlas(
        atlas,
        space=config["space"],
        resolution=config["resolution"],
        atlas_repo=config.get("atlas_repo"),
    )
    info = pd.read_csv(annotation_path)
    if vector.ndim != 1 or len(vector) != len(info):
        raise ValueError("Expected one value for each of {} parcels".format(len(info)))
    if str(atlas).casefold() not in {"fine-grained", "dense"}:
        raise ValueError("Surface visualization requires the Fine-grained atlas")

    left_label_path, right_label_path = fetch_fine_grained_surface(
        space="fs_LR",
        density="32k",
        atlas_repo=config.get("atlas_repo"),
    )
    left_labels = np.asarray(surface.load_surf_data(str(left_label_path)), dtype=int)
    right_labels = np.asarray(surface.load_surf_data(str(right_label_path)), dtype=int)
    left_vertex = np.zeros(left_labels.shape, dtype=float)
    right_vertex = np.zeros(right_labels.shape, dtype=float)
    left_mask = left_labels > 0
    right_mask = right_labels > 0
    left_vertex[left_mask] = vector[left_labels[left_mask] - 1]
    right_vertex[right_mask] = vector[right_labels[right_mask] - 1]

    vmax = float(max_value) if max_value is not None else float(np.nanmax(np.abs(vector)))
    if vmax == 0 or not np.isfinite(vmax):
        vmax = 1.0

    with tempfile.TemporaryDirectory() as temp_dir:
        rendered = visualize_surface_32k_fs_LR(
            save_path=temp_dir,
            name=title,
            dpi=200,
            mymap=cmap,
            dataL=left_vertex,
            dataR=right_vertex,
            vmax=vmax,
            threshold=0,
            darkness=0.6,
            boundary=False,
            Sym=True,
        )
    fig, axis = plt.subplots(figsize=(10, 2.5))
    axis.imshow(rendered)
    axis.axis("off")
    if output_file is not None:
        output = Path(output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(str(output))
    return fig
