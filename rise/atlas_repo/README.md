# RISE Atlas Repository

This directory contains the atlases and related files used for fMRI feature extraction and neural network training in the Regional Identity Signature Expression (RISE) framework.

## Fine-grained parcellation

`Fine-grained/` contains the 7,340-parcel cortical parcellation used as the basic spatial unit in RISE (3,670 parcels per hemisphere).

- Volume versions are provided in `MNI152NLin6Asym` and `MNI152NLin2009cAsym` spaces at the available 1, 2, and 3 mm resolutions.
- Surface versions are provided in `surf/` for `fsaverage`, `fs_LR_32k`, and `fs_LR_164k`, separately for the left (`lh`) and right (`rh`) hemispheres.
- `annot.csv` contains parcel labels, names, and centroids.
- `geodist/parcel-wise_geodist.csv` contains pairwise geodesic distances between fine-grained parcels and is used to compute the constraint loss for neural network training.

## Multi-atlas files

Eight cortical atlases provide complementary regional identity labels for each fine-grained parcel:

| Directory | Atlas | Cortical regions |
|---|---|---:|
| `DK/` | Desikan-Killiany | 68 |
| `AAL/` | Automated Anatomical Labeling | 76 |
| `HarvardOxford/` | Harvard-Oxford cortical atlas | 96 |
| `NM/` | Neuromorphometrics | 98 |
| `BN/` | Brainnetome | 210 |
| `Glasser/` | Glasser multimodal cortical atlas | 352 |
| `Schaefer200/` | Schaefer 200-parcel, 7-network atlas | 200 |
| `Schaefer400/` | Schaefer 400-parcel, 7-network atlas | 400 |


`Yeo7/` and `Yeo17/` contain the Yeo 7- and 17-network assignments used for functional connectivity extraction or network-level analyses, but not as RISE training targets.
`Buckner/` and `Tianye/` contain cerebellar and subcortical parcellations, respectively. They are used to extract time series from cerebellar regions and subcortical nuclei and to compute cortico-cerebellar and cortico-subcortical functional connectivity.


## File naming and contents

- `<Atlas>_<Space>_<Resolution>.nii.gz`: volumetric label image. Available spaces are `MNI152NLin6Asym` and `MNI152NLin2009cAsym`; the resolution suffix is `1mm`, `2mm`, or `3mm` when available.
- `annot.csv`: label metadata, including region labels, names, hemispheres, and centroid coordinates.
- `annot_sym.csv`: symmetric label metadata.
- `MNI/`: MNI template images and masks used as spatial references.

## Multi-atlas label mapping

- `multi-atlas_label_mapping.csv`: label mapping from each fine-grained parcel to the eight training atlases, together with Yeo network labels, overlap information, centroids, and normalized distance measures.
- `multi-atlas_soft_labels.csv`: smoothed soft labels used to compute the soft cross-entropy loss for neural network training.

## Python resource access

Use the public fetch functions instead of constructing paths relative to a project
checkout:

```python
from rise import (
    fetch_atlas,
    fetch_fine_grained_geodesic_distances,
    fetch_fine_grained_surface,
    fetch_fine_grained_volume,
    fetch_multi_atlas_label_mapping,
    fetch_multi_atlas_soft_labels,
)
```

The functions return absolute `pathlib.Path` objects inside the installed `rise`
package. `fetch_atlas` and `fetch_fine_grained_volume` return the volume path and
its `annot.csv`; `fetch_fine_grained_surface` returns the left and right surface
label paths.

