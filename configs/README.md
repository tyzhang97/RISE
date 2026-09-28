# RISE configuration files

Configuration files in this directory are JSON with line comments (`.jsonc`).
They remain outside the installed package so that each user can edit dataset,
output, runtime, and model parameters. Pip users should download the required
file from the GitHub `configs/` directory before running a workflow.
Preprocessing and feature presets are loaded with `rise.load_config`,
`rise.load_hctsa_config`, or `rise.load_feature_config`. Neural-network presets
are loaded by the corresponding network scripts through `--config`; do not
parse these files directly with `json.load`.

All presets that use atlas data define `atlas_repo`. Relative paths are resolved
from the directory containing the JSONC file, so `../rise/atlas_repo` works for
the standard cloned repository layout. The atlas data are not included in the
PyPI wheel. Pip users must download `atlas_repo` from the GitHub repository and
change this field to the extracted directory. RISE raises a descriptive
`FileNotFoundError` when that directory is unavailable.

## Presets

| File | Description |
| --- | --- |
| `hcp_timeseries_parameters.jsonc` | HCP-YA time-series extraction and voxelwise ReHo, ALFF, and fALFF metric calculation. |
| `hcpd_timeseries_parameters.jsonc` | HCP Development time-series extraction and voxelwise ReHo, ALFF, and fALFF metric calculation. |
| `pku6_timeseries_parameters.jsonc` | Clinical-data time-series extraction and voxelwise ReHo, ALFF, and fALFF metric calculation, using data from PKU6 site as an example. |
| `hcp_feature_parameters.jsonc` | Features extraction parameter file for HCP dataset. |
| `hcp_network_pretrain_parameters.jsonc` | Neural-network pretraining, group testing, and parcel embedding extraction. |
| `hcpd_network_finetune_parameters.jsonc` | Fine-tuning from HCP checkpoints, group testing, and parcel embedding extraction. |

## Neural-network parameters

Edit the neural-network JSONC presets instead of editing files under
`rise/neural_network`. Each preset defines input and output paths, device,
`run_name`, `models_root`, model names, architecture, optimization, validation,
scheduler, loss, and checkpoint-selection settings. The HCP-D preset additionally
defines `pretrained_model_name`.

Paths in these files are resolved relative to the directory containing the
selected configuration. The network scripts require `--config <path>`.
Explicit command-line options override values from
the selected file. `test.py` and `transform.py` should receive the same config
file used for training or fine-tuning so that paths and model settings remain
consistent. Pipeline notebooks execute installed neural-network modules with
`python -m rise.neural_network.<module>` rather than source-file paths.

## Output organization

RISE stores processed data and extracted features in a BIDS-like layout for
consistent management and downstream use:

```text
<output_root>/
├── indiv/<subject>/func/<run>/data/
└── indiv/<subject>/func/<run>/feature/
```

The `data` directory contains processed time series, voxelwise ReHo/ALFF/fALFF
maps, and intermediate FC matrices. The `feature` directory contains
parcel-level hctsa features, aligned gradients, network connectivity features,
and parcel-level ReHo/ALFF/fALFF. The `mean` directory stores features averaged
across runs. A `group/` directory is created only by workflows that write
actual group-level results; subject-level feature extraction does not create it.

Output filenames retain BIDS-like entities such as subject, run, space, atlas,
processing description, regression mode (`NGR` or `GSR`), and feature name.
For example:

```text
sub-<ID>_<run>_space-<space>_atlas-<atlas>_desc-<processing>_<NGR|GSR>_timeseries.npy
sub-<ID>_<run>_atlas-<atlas>_desc-<processing>_<NGR|GSR>_feature-<name>.npy
```

This convention follows BIDS-like naming principles while retaining the fields
needed by the RISE workflow.


## Global signal regression

`gsr_modes` controls whether feature workflows process nuisance regression
without global signal (`NGR`), with the RISE global signal (`GSR`), or both.
The feature extraction preset intentionally includes both modes.


## Model-ready Feather data

`pipeline/04_hcp_feature_integration.ipynb` defines the HCP train/test subject
files, runs, output directory, canonical feature-name resource, parcel metadata,
numeric dtype, and clipping threshold. It generates one train and one test
Feather file per run:

```text
<integration_output_root>/<run>/train_samples_parcel-level.feather
<integration_output_root>/<run>/test_samples_parcel-level.feather
<integration_output_root>/<run>/train_group-mean_parcel-level.feather
<integration_output_root>/<run>/test_group-mean_parcel-level.feather
```

Each row is one subject/parcel sample. The columns contain `Subject_ID`, the
