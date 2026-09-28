import json
import multiprocessing
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from nilearn import image, signal
from scipy import ndimage
from tqdm.auto import tqdm

from .preprocessing import (
    fill_zero_timeseries,
    load_atlas_metadata,
    strip_subject_prefix,
    validate_prepared_grid,
)
from .metrics import calculate_run_metrics, frequency_label


def timeseries_filename(
    subject_id: str,
    run_name: str,
    atlas_name: str,
    config: Mapping[str, Any],
    gsr: bool,
) -> str:
    """Construct a time-series filename using the established convention."""
    high = frequency_label(float(config["high_pass_hz"]))
    low = frequency_label(float(config["low_pass_hz"]))
    smooth = str(config["smoothing_fwhm_mm"]).zfill(2)
    detrend = "_detrend" if config["detrend"] else ""
    regression = "GSR" if gsr else "NGR"
    return (
        f"sub-{subject_id}_{run_name}_space-{config['space']}_atlas-{atlas_name}_"
        f"desc-filter{high}to{low}_sm{smooth}{detrend}_{regression}_timeseries.npy"
    )


def parameter_record(
    config: Mapping[str, Any], run_input: Mapping[str, Any]
) -> Dict[str, Any]:
    """Collect parameters stored beside an extracted run.

    Args:
        config: Full extraction configuration.
        run_input: Subject/run record, optionally containing a run-specific TR.

    Returns:
        The processing parameters needed to identify an equivalent completed run.
    """
    keys = [
        "dataset_name",
        "space",
        "resolution",
        "atlases",
        "smooth_atlases",
        "fill_zero_atlases",
        "high_pass_hz",
        "low_pass_hz",
        "smoothing_fwhm_mm",
        "detrend",
        "compute_reho_alff_falff",
        "falff_low_frequency_hz",
        "falff_high_frequency_hz",
        "gsr_modes",
        "confound_format",
        "confound_strategy",
        "fmriprep",
    ]
    record = {key: config.get(key) for key in keys}
    record["tr"] = float(run_input.get("tr", config["tr"]))
    return record


def calculate_global_signal(bold_data: np.ndarray, mask_data: np.ndarray) -> np.ndarray:
    """Calculate the global signal for each BOLD volume.

    Args:
        bold_data: Four-dimensional BOLD array.
        mask_data: Three-dimensional labeled brain-mask array.

    Returns:
        One global-signal value per time point.
    """
    n_timepoints = bold_data.shape[-1]
    global_signal = np.zeros(n_timepoints)
    for index in range(n_timepoints):
        global_signal[index] = ndimage.sum(bold_data[..., index], mask_data, index=1)
    return global_signal


def extract_atlas_timeseries(
    source_data: np.ndarray,
    atlas_data: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Average BOLD signals within atlas regions.

    Args:
        source_data: Four-dimensional raw or smoothed BOLD array.
        atlas_data: Three-dimensional integer atlas-label array.
        labels: Region labels defining the output column order.

    Returns:
        Array shaped ``(timepoints, regions)`` containing regional mean signals.
    """
    n_timepoints = source_data.shape[-1]
    timeseries = np.zeros((n_timepoints, len(labels)))
    for index in range(n_timepoints):
        timeseries[index, :] = ndimage.mean(
            source_data[..., index], atlas_data, index=labels
        )
    return timeseries


def extract_run(run_input: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract time series and voxelwise metrics for one run.

    Args:
        run_input: Mapping containing ``id``, ``nii_file``, ``output_name``, and
            optionally a run-specific ``tr``.
        config: Resolved extraction configuration.

    Returns:
        A status dictionary containing the subject ID, run name, and either
        ``completed`` or ``skipped``.

    Raises:
        ValueError: If cached spatial grids mismatch or required centroids are absent.
    """
    subject_id = strip_subject_prefix(run_input["id"])
    bold_file = Path(run_input["nii_file"])
    run_name = str(run_input["output_name"])
    tr = float(run_input.get("tr", config["tr"]))
    output_dir = (
        Path(config["output_root"]) / "indiv" / subject_id / "func" / run_name / "data"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_file = output_dir / "extraction_params.json"
    current_parameters = parameter_record(config, run_input)
    if config.get("skip_existing", False) and parameter_file.exists():
        with parameter_file.open("r", encoding="utf-8") as file:
            if json.load(file) == current_parameters:
                return {"id": subject_id, "run": run_name, "status": "skipped"}

    print(f"Loading fMRI image: {bold_file}")
    bold_img = image.load_img(bold_file)
    bold_data = np.asarray(bold_img.dataobj)
    invalid = ~np.isfinite(bold_data)
    if np.any(invalid):
        print(f"Replacing {int(invalid.sum())} NaN/Inf values with zero")
        bold_data = bold_data.copy()
        bold_data[invalid] = 0
        bold_img = image.new_img_like(bold_img, bold_data)

    started = time.time()
    smoothed_img = image.smooth_img(bold_img, fwhm=config["smoothing_fwhm_mm"])
    smoothed_data = np.asarray(smoothed_img.dataobj)
    print(f"Smoothing finished in {time.time() - started:.2f} s")

    cache_dir = Path(config["atlas_cache"])
    brain_mask_path = cache_dir / "Atlas_Mask" / "Brain_mask.nii.gz"
    brain_mask = image.load_img(brain_mask_path)
    validate_prepared_grid(bold_img, cache_dir, config["atlases"])
    global_signal = calculate_global_signal(
        bold_data, np.asarray(brain_mask.dataobj)
    )

    confounds = None
    if config.get("confound_format", "hcp") == "fmriprep":
        from nilearn.interfaces.fmriprep import load_confounds

        confound_kwargs = dict(config.get("confound_strategy", {}))
        confounds, _ = load_confounds(
            str(bold_file),
            strategy=confound_kwargs.pop("strategy", ["motion", "wm_csf"]),
            **confound_kwargs,
        )

    if config.get("compute_reho_alff_falff", True):
        for gsr in config["gsr_modes"]:
            calculate_run_metrics(
                smoothed_img,
                brain_mask_path,
                subject_id,
                run_name,
                tr,
                output_dir,
                config,
                bool(gsr),
                global_signal,
                confounds,
            )

    metadata = load_atlas_metadata(config)
    for atlas_name in config["atlases"]:
        atlas_file = cache_dir / "Atlas_Merge" / atlas_name / "atlas.nii.gz"
        labels_file = cache_dir / "Atlas_Merge" / atlas_name / "labels.csv"
        atlas_data = np.asarray(image.load_img(atlas_file).dataobj)
        labels = pd.read_csv(labels_file)["Region label"].to_numpy()
        source_data = smoothed_data if atlas_name in config["smooth_atlases"] else bold_data

        started = time.time()
        atlas_timeseries = extract_atlas_timeseries(source_data, atlas_data, labels)
        print(f"{atlas_name} extraction finished in {time.time() - started:.2f} s")

        for gsr in config["gsr_modes"]:
            confounds_for_cleaning = None
            if config.get("confound_format", "hcp") == "fmriprep":
                confounds_for_cleaning = confounds.copy()
                if gsr:
                    confounds_for_cleaning["global_signal"] = global_signal
            elif gsr:
                confounds_for_cleaning = global_signal
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"The frequency specified for the low pass filter is too "
                        r"high.*"
                    ),
                    category=UserWarning,
                )
                cleaned = signal.clean(
                    atlas_timeseries,
                    detrend=config["detrend"],
                    standardize=False,
                    low_pass=float(config["low_pass_hz"]),
                    high_pass=float(config["high_pass_hz"]),
                    t_r=tr,
                    ensure_finite=False,
                    confounds=confounds_for_cleaning,
                )
            if atlas_name in config["fill_zero_atlases"]:
                centroids = metadata[atlas_name].get("centroids")
                if centroids is None:
                    raise ValueError(f"Centroids are required to fill empty {atlas_name} parcels")
                cleaned = fill_zero_timeseries(cleaned, labels, centroids)

            filename = timeseries_filename(
                subject_id, run_name, atlas_name, config, bool(gsr)
            )
            np.save(output_dir / filename, cleaned)

    with parameter_file.open("w", encoding="utf-8") as file:
        json.dump(current_parameters, file, indent=2)
    return {"id": subject_id, "run": run_name, "status": "completed"}


def _extract_run_worker(
    run_input: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Provide a picklable worker entry point for multiprocessing."""
    return extract_run(run_input, config)


def extract_dataset(
    run_inputs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    n_processes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract a collection of subject runs.

    Args:
        run_inputs: Subject/run records accepted by :func:`extract_run`.
        config: Resolved extraction configuration.
        n_processes: Worker count. The configured value is used when omitted;
            values of one or less run serially.

    Returns:
        Completion-status dictionaries for all submitted runs.
    """
    workers = int(n_processes if n_processes is not None else config.get("n_processes", 1))
    if workers <= 1:
        return [extract_run(run, config) for run in tqdm(run_inputs)]

    worker = partial(_extract_run_worker, config=dict(config))
    results: List[Dict[str, Any]] = []
    with multiprocessing.Pool(workers) as pool:
        for result in tqdm(pool.imap_unordered(worker, run_inputs), total=len(run_inputs)):
            results.append(result)
    return results

