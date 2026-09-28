# RISE pipeline notebooks

The notebook prefix identifies the processing stage. Notebooks sharing a prefix
perform the same stage for different datasets. 
The hctsa notebooks require a Python environment with a compatible `matlab.engine` 
installation.

Install the published package in the notebook kernel environment before running
any stage:

```bash
python -m pip install "risebrain[all]==1.0.0"
```

The notebooks import the published `rise` package. They no longer prepend the
repository root to `sys.path`; `PROJECT_ROOT` is retained only for experiment data,
models, results, and JSONC configurations.

## 01. Time-series extraction

- `01_hcp_timeseries_extraction.ipynb`: discovers the four HCP-YA resting-state
  runs; extracts Fine-grained cortical, Tianye subcortical, and Buckner
  cerebellar time series; and calculates voxelwise ReHo, ALFF, and fALFF.
- `01_hcpd_timeseries_extraction.ipynb`: performs the corresponding extraction
  for the four HCP Development AP/PA runs.
- `01_pku6_timeseries_extraction.ipynb`: discovers fMRIPrep outputs for the PKU6
  clinical dataset, applies the configured preprocessing, and extracts NGR/GSR
  time series and voxelwise metrics.

## 02. hctsa extraction

- `02_hcp_hctsa_extraction.ipynb`: reads HCP cortical time series and runs the
  MATLAB hctsa workflow.

Complete stage 01 first. Successful hctsa outputs are required by the
multidimensional feature stage.

## 03. Multidimensional feature extraction

- `03_hcp_feature_extraction.ipynb`: extracts cortical FC, aligned functional
  gradients, cortico-network FC, cortical-to-subcortical/cerebellar FC,
  hctsa features, and parcel-level ReHo, ALFF, and fALFF.

For other datasets, follow the corresponding HCP processing notebooks and adapt
the dataset configuration, subject lists, feature roots, and output paths.

## 04. Feature integration

- `04_feature_integration.ipynb`: combines completed parcel features,
  standardizes them within each subject, and writes train/test sample-level and
  group-mean Feather files for each configured run.

For another dataset, use the same integration functions with that dataset's
subject lists, feature root, and output directory.

## 05-06. Neural-network pretraining and transfer

- `05_network_pretrain_and_transform.ipynb`: trains repeated HCP models, tests
  group-mean atlas classification, and exports individual train/test parcel
  embeddings. Edit `../configs/hcp_network_pretrain_parameters.jsonc` before
  running it.
- `06_network_finetune_and_transform.ipynb`: initializes every repeat from its
  matching HCP checkpoint, fine-tunes on HCP-D, tests the fine-tuned models, and
  exports HCP-D embeddings. Edit
  `../configs/hcpd_network_finetune_parameters.jsonc` before running it.

Training, testing, and embedding extraction are independent cells. They run as
installed modules with `python -m rise.neural_network.<module>`. Testing and
transform load `best_validloss_epoch.pth` by default, and `models_root` in the
selected JSONC controls where all checkpoints and predictions are stored.

## 07. RSI calculation

- `07_rsi_calculation.ipynb`: builds repeat-specific group-reference
  embeddings from HCP training data, derives the adaptive threshold from the
  training group expression matrix, and calculates single-bin and multi-bin
  RSI for the train and test splits. The notebook can optionally normalize 
  both splits by the parcel-wise training RSI mean.

## HCP-YA execution order

Run the HCP workflow in this order:

```text
  -> 01_hcp_timeseries_extraction.ipynb
  -> 02_hcp_hctsa_extraction.ipynb
  -> 03_hcp_feature_extraction.ipynb
  -> 04_feature_integration.ipynb
  -> 05_network_pretrain_and_transform.ipynb
  -> 07_rsi_calculation.ipynb
```

## HCP Development execution order

Start with `01_hcpd_timeseries_extraction.ipynb`. For hctsa extraction,
multidimensional feature extraction, and feature integration, follow stages
02-04 of the HCP workflow with HCP-D configuration, subject lists, and paths.
Then run `06_network_finetune_and_transform.ipynb` to fine-tune the HCP model
and export HCP-D embeddings.

## Clinical dataset processing

Use the HCP processing notebooks as the reference workflow for clinical
datasets. Adapt the dataset configuration, subject lists, feature roots, output
paths, and model configuration as needed.

## Output locations

Dataset paths and processing parameters are defined under `../configs`.
Time-series and feature notebooks write BIDS-like subject outputs beneath each
dataset's configured `output_root`. Integrated Feather files are written under
the dataset's `integrated_data/<run>/` directory and contain train/test data.
