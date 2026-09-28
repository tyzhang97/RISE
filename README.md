# RISE

**Regional Identity Signature Expression (RISE)** is a latent-space framework for fine-scale, individualized mapping of regional identity across the human cortex.

RISE learns generalizable latent representations of fine-grained cortical parcels from multidimensional resting-state fMRI features. By comparing an individual's parcel embeddings with group-reference embeddings, RISE derives the **Regional Signature Index (RSI)**, an individualized whole-cortex functional phenotype that quantifies the expression of canonical regional identities.

![Overview of the RISE framework](figure/Fig1.png)

*Figure 1. Overview of the RISE framework.*

## Why RISE?

- Regional identity has been well established at the transcriptomic and cellular levels, but the reliance on post-mortem tissue and surgical specimens limits its non-invasive assessment in individual living brains. How to detect and quantify individualized regional identity throughout the cortex in vivo therefore remains largely unexplored.

- Resting-state fMRI provides complementary functional information about cortical organization. Functional gradients, cortico-network and cortico-subcortical connectivity, and intrinsic temporal dynamics together form a multifaceted imaging profile that captures the functional properties of each cortical region. These complementary features collectively establish the basis for an imaging-derived representation of regional identity, rather than describing only one isolated dimension of cortical function.

- RISE learns an integrative latent representation from these multifaceted functional profiles and, to our knowledge, provides the first framework for in vivo detection and quantitative measurement of regional identity at the individual level.

## Key advantages

- **Fine-grained and atlas-agnostic quantification**: RISE learns from integrative regional identity priors across multiple atlases and represents the cortex with 7,340 fine-grained parcels.
- **Robustness to individual topological shifts**: RSI quantifies the aggregate expression strength of a regional identity, remaining sensitive to reduced expression while robust to shifts in the locations of functional regions.
- **Generalizability and transferability**: a pretrained model can be applied to independent subjects and fine-tuned for previously unseen datasets, including clinical cohorts.

## What can RISE do?

RISE provides a comprehensive pipeline from parcel-level time-series extraction through feature computation, neural-network training or fine-tuning, and RSI calculation:

1. Extract parcel-level time series and classical local metrics from preprocessed resting-state fMRI.
2. Compute multidimensional parcel-level rs-fMRI features, including functional connectivity, aligned gradients, cortico-network connectivity, cortico-subcortical/cerebellar connectivity, hctsa temporal dynamics, ReHo, ALFF, and fALFF.
3. Integrate the feature profiles into model-ready parcel tables.
4. Train a neural network to extract regional identity representations using multi-atlas supervision and auxiliary optimization objectives, or fine-tune a pretrained model on an unseen dataset.
5. Export individual parcel embeddings, construct group-reference embeddings for each parcel, and calculate individualized RSI maps.

## Further reading

For further details about RISE, please refer to our manuscript:

> **Preprint:** [Coming soon](#)



## Installation

You may either install the RISE package in an existing Python environment or create a new environment for the complete project workflow.

### Install the published package from PyPI

The package is distributed on PyPI under the name `risebrain` and imported in Python as `rise`.

```bash
python -m pip install risebrain
```

This installs the RISE package and its required Python dependencies.

> **Atlas resources must be downloaded separately.** Due to PyPI's package-size
> limit, we cannot bundle all resources in the published package. If you install
> only `risebrain` into your own environment, you must also download the complete
> [`rise/atlas_repo/`](rise/atlas_repo/) directory from this repository and set
> `atlas_repo` in the relevant JSONC presets under [`configs/`](configs/) to its
> local path. Preserve the directory structure of the downloaded resources.

For example, set the following field in the selected configuration:

```jsonc
"atlas_repo": "/absolute/path/to/atlas_repo"
```

If you have downloaded the repository with its atlas resources, use
`"atlas_repo": "../rise/atlas_repo"` in a preset under `configs/`. Relative
paths are resolved from the configuration file's directory, not the current
working directory. See [`configs/README.md`](configs/README.md) for configuration
instructions and [`rise/atlas_repo/README.md`](rise/atlas_repo/README.md) for the
resource layout.


```python
import rise

print(rise.__version__)
```

Dataset inputs, trained models, results, MATLAB, MATLAB Engine for Python, hctsa, and MATLAB toolboxes are external resources and are not bundled in the wheel.

### Create the project environment from `environment.yml`

From the repository root:

```bash
conda env create -f environment.yml
conda activate rise_env
```

The project environment installs the published package and the complete analysis and notebook dependencies. It does not download the external atlas resources; download and configure them as described above. A pip equivalent is:

```bash
python -m pip install -r requirements.txt
```

> **Note:** The hctsa tool must be installed separately from its official repository. See [`hctsa_repo/README.md`](hctsa_repo/README.md) for the required setup.

The hctsa workflow additionally requires a compatible MATLAB Engine installation and the hctsa repository under `hctsa_repo/`. See [`hctsa_repo/README.md`](hctsa_repo/README.md).

## Usage

The workflow is organized into independent stages. RISE operates on fMRI data that have already undergone preprocessing; for example, preprocessing can be performed with [fMRIPrep](https://fmriprep.org/). RISE provides an interface for fMRIPrep derivatives through `build_fmriprep_run_inputs`. Dataset paths and processing parameters are defined in [`configs/`](configs/), and the executable notebooks are documented in [`pipeline/README.md`](pipeline/README.md).

> **Before running any workflow, modify the selected configuration for your own
> dataset and computing environment.** Update dataset and output paths, subject
> lists, atlas/cache locations, resource settings, and model
> parameters as appropriate. Do not run a preset unchanged unless its paths and
> assumptions match your data. Read [`configs/README.md`](configs/README.md)
> first; it explains the available JSONC presets, path resolution, required
> configuration fields, and neural-network command-line overrides.

### 1. Load configuration

Select and edit a JSONC preset in [`configs/`](configs/). The configuration loader resolves paths, discovers subjects and runs, prepares/resamples atlas images, and creates the brain-mask cache. The main public interfaces are:

```python
from rise import load_config, read_subject_ids, prepare_atlases

config = load_config("configs/hcp_timeseries_parameters.jsonc")
subjects = read_subject_ids("data/HCP/info/hcp_ids.csv")
```

### 2. Time-series extraction

Build run inputs with the dataset-specific discovery function and extract parcel time series and voxelwise metrics:

```python
from rise import build_hcp_run_inputs, extract_dataset

runs = build_hcp_run_inputs(subjects, config)
extract_dataset(runs, config)
```

The same API supports HCP-D and fMRIPrep derivatives through `build_hcpd_run_inputs` and `build_fmriprep_run_inputs`. Outputs use a BIDS-like layout under `indiv/<subject>/func/<run>/data/`. See the stage-01 notebooks in [`pipeline/`](pipeline/).

### 3. Feature calculation and integration

After time-series extraction, run hctsa where required and compute the multidimensional feature families. Use the feature configuration corresponding to the dataset, and update its input/output paths and feature settings before execution. RISE provides both category-specific feature extraction functions and the one-step `extract_all_features` entry point:

```python
from rise import extract_all_features

outputs = extract_all_features(subjects, feature_config, modes=["NGR", "GSR"])
```

See [`pipeline/README.md`](pipeline/README.md) and [`configs/README.md`](configs/README.md).

### 4. Neural network training and transfer

After editing the selected pretraining or fine-tuning configuration, train, test, and export embeddings through the executable modules:

```bash
python -m rise.neural_network.train \
    --config configs/hcp_network_pretrain_parameters.jsonc
python -m rise.neural_network.test \
    --config configs/hcp_network_pretrain_parameters.jsonc
python -m rise.neural_network.transform \
    --config configs/hcp_network_pretrain_parameters.jsonc
```

For fine-tuning, see [`pipeline/06_network_finetune_and_transform.ipynb`](pipeline/06_network_finetune_and_transform.ipynb) and `configs/hcpd_network_finetune_parameters.jsonc`. Training and transfer details are summarized in [`pipeline/README.md`](pipeline/README.md).

### 5. RSI calculation

The RSI workflow loads repeated train/test embeddings, constructs parcel-wise group-reference embeddings, computes individual expression matrices, derives an adaptive threshold, and aggregates above-threshold expression into single-bin or multi-bin RSI:

```python
from rise import compute_training_rsi, compute_test_rsi
```

The executable workflow is [`pipeline/07_rsi_calculation.ipynb`](pipeline/07_rsi_calculation.ipynb), and the implementation is [`rise/rsi.py`](rise/rsi.py). RSI outputs are stored under `results/rsi/`.

### 6. Downstream analyses

The repository provides notebooks for model validation and downstream analyses:

- [Evaluation](analysis/evaluation/README.md): model and RSI validation, including classification accuracy and fingerprinting analysis.
- [Development](analysis/development/README.md): developmental analyses based on RISE representations and RSI.
- [Behavior](analysis/behavior/README.md): RSI-based individual behavioral prediction and brain-behavior association maps.
- [Clinical](analysis/clinical/README.md): clinical generalization of RISE, including case-control Cohen's *d* maps.

## Repository layout

```text
rise/                 Core preprocessing, feature, neural-network, and RSI package
configs/              JSONC dataset and model presets
pipeline/              End-to-end executable notebooks
analysis/              Evaluation, development, behavioral, and clinical analyses
rise/atlas_repo/       Atlas and multi-atlas resources downloaded from this repository
hctsa_repo/            External MATLAB hctsa dependency
models/                Trained model checkpoints
results/               Derived predictions, embeddings, and RSI outputs
data/                  Dataset metadata, subject lists, and processed data
```

## Citation

If you use RISE, please cite the accompanying manuscript. The citation and preprint link will be added here when available.

## Support

For questions, please email the author ([tyzhang@ibp.ac.cn](mailto:tyzhang@ibp.ac.cn)).
