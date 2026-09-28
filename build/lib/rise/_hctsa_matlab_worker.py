"""Internal bridge between the RISE Python environment and MATLAB Engine."""

from __future__ import print_function

import io
import json
import sys
import time

import matlab.engine


def main(manifest_path):
    """Run the configured hctsa jobs through the MATLAB Engine.

    Args:
        manifest_path: JSON manifest written by ``process_hctsa_subject``.

    Returns:
        None. MATLAB writes feature MAT files and this worker writes one
        completion marker per successfully processed run.
    """
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)

    engine = None
    for attempt in range(1, 4):
        try:
            engine = matlab.engine.start_matlab()
            break
        except matlab.engine.EngineError:
            if attempt == 3:
                raise
            delay = 10 * attempt
            print(
                "MATLAB Engine startup failed; retrying in {} seconds "
                "(attempt {}/3).".format(delay, attempt),
                file=sys.stderr,
            )
            time.sleep(delay)

    try:
        engine.addpath(manifest["matlab_function_dir"], nargout=0)
        function = getattr(engine, manifest["matlab_function"])
        for job in manifest["jobs"]:
            for label in job["timeseries_labels"]:
                function(
                    manifest["do_pool"],
                    manifest["n_workers"],
                    label,
                    manifest["feature_type"],
                    job["input_dir"],
                    job["initial_dir"],
                    job["output_dir"],
                    manifest["hctsa_root"],
                    manifest["mops_file"],
                    manifest["ops_file"],
                    nargout=0,
                    stdout=io.StringIO(),
                )
            with open(job["done_file"], "w") as handle:
                handle.write("Processing complete")
            print("Completed {subject} {run} {gsr_label}".format(**job))
    finally:
        engine.quit()


if __name__ == "__main__":
    main(sys.argv[1])
