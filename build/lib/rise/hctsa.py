import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.io import savemat

from .metrics import frequency_label
from .fetch import _fetch_matlab_function_directory
from .preprocessing import load_config, strip_subject_prefix

PathLike = Union[str, os.PathLike]
HCTSA_FEATURE_TYPE = "all"
_HCTSA_PATH_KEYS = (
    "hctsa_root",
    "hctsa_mops_file",
    "hctsa_ops_file",
)


def load_hctsa_config(
    config_path: PathLike,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load an hctsa configuration and resolve its additional path fields."""
    path = Path(config_path).expanduser().resolve()
    config = load_config(path, overrides=overrides)
    for key in _HCTSA_PATH_KEYS:
        if config.get(key):
            value = Path(config[key]).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            config[key] = str(value.resolve())
    config["matlab_function_dir"] = str(_fetch_matlab_function_directory())
    return config


def hctsa_timeseries_suffix(
    config: Mapping[str, Any],
    gsr_label: str,
    atlas_name: Optional[str] = None,
) -> str:
    """Build the suffix used to select one atlas/regression time series."""
    label = gsr_label.upper()
    if label not in {"GSR", "NGR"}:
        raise ValueError(f"gsr_label must be GSR or NGR, got {gsr_label!r}")
    atlas = atlas_name or str(config["atlas_name"])
    high = frequency_label(float(config["high_pass_hz"]))
    low = frequency_label(float(config["low_pass_hz"]))
    smooth = str(config["smoothing_fwhm_mm"]).zfill(2)
    detrend = "_detrend" if config["detrend"] else ""
    return (
        f"atlas-{atlas}_desc-filter{high}to{low}_sm{smooth}"
        f"{detrend}_{label}_timeseries.npy"
    )


def build_hctsa_run_inputs(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    gsr_modes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Discover run-level time-series inputs under the RISE output tree."""
    output_root = Path(config["output_root"])
    modes = list(gsr_modes or config.get("gsr_modes", ["NGR"]))
    records: List[Dict[str, Any]] = []

    for subject in subjects:
        subject_id = strip_subject_prefix(subject)
        func_dir = output_root / "indiv" / subject_id / "func"
        if not func_dir.is_dir():
            continue
        for run_dir in sorted(path for path in func_dir.iterdir() if path.is_dir()):
            if run_dir.name == "mean":
                continue
            data_dir = run_dir / "data"
            for mode in modes:
                label = str(mode).upper()
                suffix = hctsa_timeseries_suffix(config, label)
                files = sorted(data_dir.glob(f"*{suffix}"))
                if files:
                    records.append(
                        {
                            "id": subject_id,
                            "run": run_dir.name,
                            "gsr_label": label,
                            "data_dir": str(data_dir),
                            "timeseries_files": [str(path) for path in files],
                        }
                    )
    return records


def _hctsa_directories(record: Mapping[str, Any]) -> Dict[str, Path]:
    """Return the historical hctsa input, initialization, and output folders.

    Args:
        record: Run record containing the atlas time-series ``data_dir``.

    Returns:
        Mapping with the run-level hctsa root, MAT input directory, MATLAB
        initialization directory, and feature output directory.
    """
    root = Path(record["data_dir"]) / "hctsa"
    return {
        "root": root,
        "input": root / "inpMat",
        "initial": root / "initMat",
        "output": root / "outFeatures",
    }


def hctsa_done_file(record: Mapping[str, Any]) -> Path:
    """Return the historical completion-marker path for one run and mode."""
    return (
        _hctsa_directories(record)["output"]
        / f"{record['gsr_label']}_Done.txt"
    )


def prepare_hctsa_run(
    record: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Write historical ``data``/``labels`` MAT inputs for one run."""
    directories = _hctsa_directories(record)
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    (directories["output"] / HCTSA_FEATURE_TYPE).mkdir(parents=True, exist_ok=True)

    labels = []
    for timeseries_file in record["timeseries_files"]:
        path = Path(timeseries_file)
        data = np.load(path)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D time-series array, got {data.shape}: {path}")
        region_labels = list(map(str, range(data.shape[1])))
        label = path.stem
        savemat(
            directories["input"] / f"{label}_inp.mat",
            {"data": data, "labels": region_labels},
        )
        labels.append(label)

    return {
        "subject": record["id"],
        "run": record["run"],
        "gsr_label": record["gsr_label"],
        "timeseries_labels": labels,
        "input_dir": str(directories["input"]),
        "initial_dir": str(directories["initial"]),
        "output_dir": str(directories["output"]),
        "done_file": str(hctsa_done_file(record)),
    }


def validate_hctsa_environment(config: Mapping[str, Any]) -> Dict[str, str]:
    """Validate hctsa paths and MATLAB Engine in the current Python environment."""
    required = {
        "hctsa_root": Path(config["hctsa_root"]),
        "hctsa_mops_file": Path(config["hctsa_mops_file"]),
        "hctsa_ops_file": Path(config["hctsa_ops_file"]),
        "matlab_function_dir": Path(config["matlab_function_dir"]),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing hctsa dependencies:\n" + "\n".join(missing))

    command = [
        sys.executable,
        "-c",
        "import matlab.engine; print('MATLAB Engine import OK')",
    ]
    completed = subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    return {
        "matlab_engine": completed.stdout.strip(),
        "python": sys.executable,
        "hctsa_root": str(required["hctsa_root"]),
        "operations": str(required["hctsa_ops_file"]),
    }


def process_hctsa_subject(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Prepare and compute all pending hctsa runs for one subject."""
    if not records:
        raise ValueError("At least one run record is required")
    subject_ids = {str(record["id"]) for record in records}
    if len(subject_ids) != 1:
        raise ValueError("All records passed to process_hctsa_subject must share one subject")

    skip_existing = bool(config.get("skip_existing", True))
    pending = [
        record
        for record in records
        if not (skip_existing and hctsa_done_file(record).exists())
    ]
    subject_id = next(iter(subject_ids))
    if not pending:
        return {"id": subject_id, "status": "skipped", "runs": 0}

    started = time.time()
    jobs = [prepare_hctsa_run(record, config) for record in pending]
    manifest = {
        "jobs": jobs,
        "matlab_function": str(config.get("matlab_function", "ts_hctsa_workers")),
        "matlab_function_dir": str(config["matlab_function_dir"]),
        "hctsa_root": str(config["hctsa_root"]),
        "mops_file": str(config["hctsa_mops_file"]),
        "ops_file": str(config["hctsa_ops_file"]),
        "feature_type": HCTSA_FEATURE_TYPE,
        "do_pool": bool(config.get("matlab_do_pool", False)),
        "n_workers": int(config.get("matlab_workers", 16)),
    }
    temp_dir = Path(config.get("temporary_directory", tempfile.gettempdir()))
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = temp_dir / f"rise_hctsa_{subject_id}_{os.getpid()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    worker_script = Path(__file__).resolve().parent / "_hctsa_matlab_worker.py"
    try:
        completed = subprocess.run(
            [sys.executable, str(worker_script), str(manifest_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode != 0:
            raise RuntimeError(
                "hctsa MATLAB worker failed for subject {} (exit code {}).\n{}"
                .format(subject_id, completed.returncode, completed.stdout or "")
            )
    finally:
        if manifest_path.exists():
            manifest_path.unlink()

    return {
        "id": subject_id,
        "status": "completed",
        "runs": len(jobs),
        "minutes": (time.time() - started) / 60,
    }


def extract_hctsa_dataset(
    run_inputs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    n_jobs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Compute hctsa features in parallel across subjects."""
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in run_inputs:
        grouped[str(record["id"])].append(record)
    subject_records = [grouped[key] for key in sorted(grouped)]
    workers = int(n_jobs if n_jobs is not None else config.get("n_jobs", 1))
    if workers == 1:
        return [process_hctsa_subject(records, config) for records in subject_records]
    # Each task launches its own Python/MATLAB worker. Threads avoid spawning that
    # worker from inside a loky child process and do not limit MATLAB computation.
    return Parallel(n_jobs=workers, prefer="threads")(
        delayed(process_hctsa_subject)(records, config) for records in subject_records
    )


def find_missing_hctsa_runs(
    subjects: Iterable[Union[str, int]],
    config: Mapping[str, Any],
    gsr_modes: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    """List subject/run/mode combinations without historical done markers."""
    output_root = Path(config["output_root"])
    modes = [str(mode).upper() for mode in (gsr_modes or config.get("gsr_modes", ["NGR"]))]
    missing: List[Dict[str, str]] = []
    for subject in subjects:
        subject_id = strip_subject_prefix(subject)
        func_dir = output_root / "indiv" / subject_id / "func"
        if not func_dir.is_dir():
            continue
        for run_dir in sorted(path for path in func_dir.iterdir() if path.is_dir()):
            if run_dir.name == "mean":
                continue
            for mode in modes:
                marker = run_dir / "data" / "hctsa" / "outFeatures" / f"{mode}_Done.txt"
                if not marker.exists():
                    missing.append({"id": subject_id, "run": run_dir.name, "gsr_label": mode})
    return missing
