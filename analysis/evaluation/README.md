# RISE and RSI Analyses and Validation

This directory contains notebooks for analyzing and validating the RISE (Regional Identity Signature Expression) framework and RSI (Regional Specificity Index). These notebooks evaluate and visualize previously generated model predictions, individual embeddings, and RSI results.

## Analyses

### 1. Generalization accuracy on an independent test set

- `hcp_classification_accuracy.ipynb`
- Loads parcel-level labels predicted by the pretrained HCP models on the independent test set, calculates classification accuracy for each cortical atlas, and stratifies the results by percentile ranges of the distance between each parcel and its target atlas centroid.
- Output: Group-level classification accuracy results and a grouped bar plot consistent with the original analysis.

### 2. Fingerprinting (individual identification accuracy)

- `fingerprinting.ipynb`
- Evaluates the accuracy of RISE representations in individual identification, including the matching procedure, summary of identification results, and visualization.

### 3. Association between the RSI inter-individual variation map and cortical organization

- `inter-subject_RSI_variation.ipynb`
- Examines the spatial association between the RSI inter-individual variation map and cortical organization principles, and presents the corresponding statistical results and visualizations.