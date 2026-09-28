# RISE behavioral analyses

This directory contains the HCP individual behavioral prediction and
brain-behavior association analyses based on RSI metrics.

### 1. Individual behavioral prediction

- `behavior_predict.ipynb`
- Use cross-validated ridge regression to predict individual behavioral phenotypes.


### 2. Brain-behavior association

- `brain_behavior_association.ipynb`
- Calculate the correlation between RSI and behavioral scores to obtain a whole-cortex brain-behavior association map.


### 3. Validation using task-evoked activations
- `task_validation.ipynb`
- Utilize the individual task-specific contrast maps provided by the HCP-YA dataset to validate whether the inter-individual differences reflected by the RSI could be supported by actual activation differences during related tasks.

