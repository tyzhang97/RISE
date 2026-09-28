"""ReHo, ALFF, and fALFF calculations for volumetric fMRI data."""

import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from nilearn import image, masking
from scipy.fft import fft

PathLike = Union[str, os.PathLike]


def mean_normalize_nonzero(input_path: PathLike, output_path: PathLike) -> None:
    """Mean-normalize the nonzero voxels of a NIfTI image.

    Args:
        input_path: Source NIfTI image.
        output_path: Destination for the normalized NIfTI image.

    Returns:
        None. The normalized image is written to ``output_path``.
    """
    img = nib.load(str(input_path))
    data = img.get_fdata()
    nonzero_mask = data != 0
    normalized = np.zeros_like(data)
    normalized[nonzero_mask] = data[nonzero_mask] / data[nonzero_mask].mean()
    nib.save(nib.Nifti1Image(normalized, img.affine, img.header), str(output_path))


def next_power_of_two(value: int) -> int:
    """Return the smallest power of two greater than or equal to ``value``."""
    result = 1
    while result < value:
        result *= 2
    return result


def calculate_alff_falff(
    timeseries: np.ndarray,
    min_low_frequency: float,
    max_low_frequency: float,
    tr: float,
) -> Tuple[float, float]:
    """Calculate ALFF and fALFF for one voxel.

    Args:
        timeseries: One-dimensional voxel time series.
        min_low_frequency: Lower bound of the ALFF frequency band in Hz.
        max_low_frequency: Upper bound of the ALFF frequency band in Hz.
        tr: Repetition time in seconds.

    Returns:
        A tuple containing the ALFF value and fALFF ratio.
    """
    n_timepoints = len(timeseries)
    sampling_frequency = 1 / tr
    n_fft = next_power_of_two(n_timepoints)
    magnitude = 2 * np.abs(fft(timeseries, n=n_fft)) / n_timepoints
    frequencies = sampling_frequency / 2 * np.linspace(0, 1, int(n_fft / 2 + 1))
    low_indices = np.where(
        (min_low_frequency <= frequencies) & (frequencies <= max_low_frequency)
    )
    alff = float(np.sum(magnitude[low_indices]))
    total = float(np.sum(magnitude))
    falff = alff / total if total > 0 else 0.0
    return alff, falff


def calculate_alff_falff_image(
    input_img,
    min_low_frequency: float,
    max_low_frequency: float,
    tr: float,
    mask_path: PathLike,
):
    """Calculate voxelwise ALFF and fALFF images.

    Args:
        input_img: Four-dimensional niimg-like BOLD image.
        min_low_frequency: Lower bound of the ALFF frequency band in Hz.
        max_low_frequency: Upper bound of the ALFF frequency band in Hz.
        tr: Repetition time in seconds.
        mask_path: Brain-mask NIfTI file.

    Returns:
        A tuple containing the ALFF image and fALFF image.
    """
    mask_img = image.load_img(mask_path)
    if not np.array_equal(input_img.affine, mask_img.affine):
        mask_img = image.resample_to_img(mask_img, input_img, interpolation="nearest")

    voxel_timeseries = masking.apply_mask(input_img, mask_img)
    alff_values = np.zeros(voxel_timeseries.shape[1])
    falff_values = np.zeros(voxel_timeseries.shape[1])
    for index in range(voxel_timeseries.shape[1]):
        alff_values[index], falff_values[index] = calculate_alff_falff(
            voxel_timeseries[:, index], min_low_frequency, max_low_frequency, tr
        )
    return masking.unmask(alff_values, mask_img), masking.unmask(falff_values, mask_img)


def calculate_reho(
    input_file: PathLike,
    output_file: PathLike,
    mask_file: Optional[PathLike] = None,
    neighborhood: str = "vertices",
):
    """Calculate ReHo with AFNI's ``3dReHo`` through Nipype.

    Args:
        input_file: Filtered four-dimensional BOLD NIfTI file.
        output_file: Destination ReHo NIfTI file.
        mask_file: Optional brain-mask NIfTI file.
        neighborhood: AFNI neighborhood definition, such as ``vertices`` or
            ``cubes``.

    Returns:
        The Nipype interface result returned by ``afni.ReHo.run``.
    """
    from nipype.interfaces import afni

    reho = afni.ReHo()
    reho.inputs.in_file = str(input_file)
    reho.inputs.out_file = str(output_file)
    reho.inputs.neighborhood = neighborhood
    if mask_file is not None:
        reho.inputs.mask_file = str(mask_file)
    return reho.run()


def frequency_label(value: float) -> str:
    """Format a frequency value as used in the original output filenames."""
    text = str(value)
    return text.split(".", maxsplit=1)[1] if "." in text else text


def metric_output_stem(
    subject_id: str,
    run_name: str,
    config: Mapping[str, Any],
    gsr: bool,
) -> str:
    """Construct the shared filename stem for voxelwise metric outputs."""
    high = frequency_label(float(config["high_pass_hz"]))
    low = frequency_label(float(config["low_pass_hz"]))
    smooth = str(config["smoothing_fwhm_mm"]).zfill(2)
    detrend = "_detrend" if config["detrend"] else ""
    regression = "GSR" if gsr else "NGR"
    return (
        f"sub-{subject_id}_{run_name}_space-{config['space']}_"
        f"desc-filter{high}to{low}_sm{smooth}{detrend}_{regression}"
    )


def calculate_run_metrics(
    bold_img,
    brain_mask_path: Path,
    subject_id: str,
    run_name: str,
    tr: float,
    output_dir: Path,
    config: Mapping[str, Any],
    gsr: bool,
    global_signal: np.ndarray,
    confounds: Optional[Any] = None,
) -> None:
    """Calculate voxelwise metrics for one run and GSR mode.

    Args:
        bold_img: Smoothed four-dimensional niimg-like BOLD image.
        brain_mask_path: Brain-mask NIfTI file matching the BOLD grid.
        subject_id: Subject identifier without a ``sub-`` prefix.
        run_name: Output run identifier.
        tr: Repetition time in seconds.
        output_dir: Directory in which metric images are saved.
        config: Extraction and frequency parameters.
        gsr: Whether to regress the global signal.
        global_signal: One-dimensional global-signal confound.
        confounds: Optional fMRIPrep confound table used for both NGR and GSR;
            the global signal is added to a copy for GSR when provided.

    Returns:
        None. ReHo, ALFF, fALFF, and mean-normalized images are saved to disk.
    """
    stem = metric_output_stem(subject_id, run_name, config, gsr)
    outputs = [
        output_dir / f"{stem}_REHO.nii.gz",
        output_dir / f"{stem}_REHO_norm.nii.gz",
        output_dir / f"{stem}_ALFF.nii.gz",
        output_dir / f"{stem}_ALFF_norm.nii.gz",
        output_dir / f"{stem}_fALFF.nii.gz",
        output_dir / f"{stem}_fALFF_norm.nii.gz",
    ]
    if config.get("skip_existing", False) and all(path.exists() for path in outputs):
        return

    metric_confounds = confounds.copy() if confounds is not None else None
    if gsr:
        if metric_confounds is None:
            metric_confounds = global_signal
        else:
            metric_confounds["global_signal"] = global_signal
    started = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"The frequency specified for the low pass filter is too "
                r"high.*"
            ),
            category=UserWarning,
        )
        filtered_img = image.clean_img(
            bold_img,
            detrend=config["detrend"],
            standardize=False,
            confounds=metric_confounds,
            low_pass=float(config["low_pass_hz"]),
            high_pass=float(config["high_pass_hz"]),
            t_r=tr,
        )
    temp_dir = Path(config.get("temporary_directory", tempfile.gettempdir()))
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{stem}_bold.nii.gz"
    nib.save(filtered_img, temp_file)
    try:
        if outputs[0].exists():
            outputs[0].unlink()
        calculate_reho(temp_file, outputs[0], brain_mask_path)
    finally:
        if temp_file.exists():
            temp_file.unlink()
    mean_normalize_nonzero(outputs[0], outputs[1])
    print(f"ReHo ({'GSR' if gsr else 'NGR'}) finished in {time.time() - started:.2f} s")

    # ALFF and fALFF use detrended but unfiltered BOLD data, matching the source workflow.
    started = time.time()
    unfiltered_img = image.clean_img(
        bold_img,
        detrend=config["detrend"],
        standardize=False,
        confounds=metric_confounds,
        t_r=tr,
    )
    alff_img, falff_img = calculate_alff_falff_image(
        unfiltered_img,
        float(config["falff_low_frequency_hz"]),
        float(config["falff_high_frequency_hz"]),
        tr,
        brain_mask_path,
    )
    nib.save(alff_img, outputs[2])
    mean_normalize_nonzero(outputs[2], outputs[3])
    nib.save(falff_img, outputs[4])
    mean_normalize_nonzero(outputs[4], outputs[5])
    print(f"ALFF/fALFF ({'GSR' if gsr else 'NGR'}) finished in {time.time() - started:.2f} s")

