"""Resolve atlas resources from a user-managed RISE atlas repository."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union


PathLike = Union[str, os.PathLike]
_PACKAGE_ROOT = Path(__file__).resolve().parent
_DEFAULT_ATLAS_REPO = _PACKAGE_ROOT / "atlas_repo"
_FEATURE_RESOURCES = _PACKAGE_ROOT / "feature_resources"
_MATLAB_FUNCTIONS = _PACKAGE_ROOT / "matlab"
_ATLAS_ALIASES = {"dense": "Fine-grained"}
_SPACE_NAMES = {
    "mni152nlin6asym": "MNI152NLin6Asym",
    "mni152nlin2009casym": "MNI152NLin2009cAsym",
}


def _atlas_repo(atlas_repo: Optional[PathLike] = None) -> Path:
    if atlas_repo is not None:
        root = Path(atlas_repo).expanduser().resolve()
    else:
        candidates = [_DEFAULT_ATLAS_REPO]
        search_roots = [Path.cwd(), *Path.cwd().parents]
        candidates.extend(parent / "rise" / "atlas_repo" for parent in search_roots)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate.resolve()
        root = _DEFAULT_ATLAS_REPO
    if root.is_dir():
        return root
    raise FileNotFoundError(
        "RISE atlas repository not found: {}. Download the atlas_repo data from "
        "the RISE GitHub repository and set 'atlas_repo' in the JSONC parameter "
        "file to that directory (for a cloned project, use "
        "'../rise/atlas_repo' from the configs directory).".format(root)
    )


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(description, path))
    return path


def _fetch_atlas_repository_file(
    resource: PathLike,
    atlas_repo: Optional[PathLike] = None,
    description: str = "Atlas repository resource",
) -> Path:
    """Resolve an absolute or repository-relative internal resource path."""
    path = Path(resource).expanduser()
    if not path.is_absolute():
        path = _atlas_repo(atlas_repo) / path
    return _require_file(path.resolve(), description)


def _fetch_feature_resource(filename: str) -> Path:
    """Return one packaged feature resource by filename."""
    return _require_file(_FEATURE_RESOURCES / filename, "Feature resource")


def _fetch_matlab_function_directory() -> Path:
    """Return the directory containing the packaged MATLAB helper."""
    helper = _require_file(
        _MATLAB_FUNCTIONS / "ts_hctsa_workers.m", "MATLAB hctsa helper"
    )
    return helper.parent


def _atlas_name(root: Path, atlas_name: str) -> str:
    requested = str(atlas_name).strip()
    requested = _ATLAS_ALIASES.get(requested.casefold(), requested)
    directories = {
        path.name.casefold(): path.name for path in root.iterdir() if path.is_dir()
    }
    resolved = directories.get(requested.casefold())
    if resolved is None:
        available = sorted(path.name for path in root.iterdir() if path.is_dir())
        raise ValueError(
            "Unknown atlas {!r}. Available atlas directories: {}".format(
                atlas_name, ", ".join(available)
            )
        )
    return resolved


def _space_name(space: str) -> str:
    requested = str(space).strip()
    resolved = _SPACE_NAMES.get(requested.casefold())
    if resolved is None:
        raise ValueError("space must be MNI152NLin6Asym or MNI152NLin2009cAsym")
    return resolved


def _resolution_name(resolution: Union[str, int]) -> str:
    value = str(resolution).strip().lower()
    if value.startswith("res-"):
        value = value[4:]
    if value.endswith("mm"):
        value = value[:-2]
    if not value.isdigit() or int(value) < 1:
        raise ValueError("resolution must be a positive millimeter value")
    return "{}mm".format(int(value))


def _annotation_file(atlas_dir: Path) -> Path:
    for filename in ("annot.csv", "Annot.csv"):
        path = atlas_dir / filename
        if path.is_file():
            return path
    raise FileNotFoundError("Atlas annotation not found in {}".format(atlas_dir))


def fetch_atlas(
    atlas_name: str,
    space: str = "MNI152NLin6Asym",
    resolution: Union[str, int] = "2mm",
    atlas_repo: Optional[PathLike] = None,
) -> Tuple[Path, Path]:
    """Return a volumetric atlas and annotation from the configured repository."""
    root = _atlas_repo(atlas_repo)
    name = _atlas_name(root, atlas_name)
    atlas_dir = root / name
    atlas_path = atlas_dir / "{}_{}_{}.nii.gz".format(
        name, _space_name(space), _resolution_name(resolution)
    )
    return _require_file(atlas_path, "Atlas image"), _annotation_file(atlas_dir)


def fetch_fine_grained_volume(
    space: str = "MNI152NLin6Asym",
    resolution: Union[str, int] = "2mm",
    atlas_repo: Optional[PathLike] = None,
) -> Tuple[Path, Path]:
    """Return the Fine-grained volume atlas and parcel annotation CSV."""
    return fetch_atlas(
        "Fine-grained", space=space, resolution=resolution, atlas_repo=atlas_repo
    )


def fetch_fine_grained_surface(
    space: str = "fs_LR",
    density: str = "32k",
    atlas_repo: Optional[PathLike] = None,
) -> Tuple[Path, Path]:
    """Return left and right Fine-grained surface label files."""
    root = _atlas_repo(atlas_repo)
    surface_dir = root / "Fine-grained" / "surf"
    normalized_space = str(space).strip().replace("-", "_").casefold()
    if normalized_space in {"fslr", "fs_lr"}:
        normalized_density = str(density).strip().lower()
        if normalized_density not in {"32k", "164k"}:
            raise ValueError("fs_LR density must be 32k or 164k")
        suffix = "fs_LR_{}.gii".format(normalized_density)
    elif normalized_space == "fsaverage":
        suffix = "fsaverage.annot"
    else:
        raise ValueError("surface space must be fs_LR or fsaverage")
    return (
        _require_file(
            surface_dir / "lh.fine-grained.{}".format(suffix),
            "Left-hemisphere Fine-grained surface",
        ),
        _require_file(
            surface_dir / "rh.fine-grained.{}".format(suffix),
            "Right-hemisphere Fine-grained surface",
        ),
    )


def fetch_multi_atlas_label_mapping(
    atlas_repo: Optional[PathLike] = None,
) -> Path:
    """Return the hard-label mapping for all training atlases."""
    return _require_file(
        _atlas_repo(atlas_repo) / "multi-atlas_label_mapping.csv",
        "Multi-atlas label mapping",
    )


def fetch_multi_atlas_soft_labels(
    atlas_repo: Optional[PathLike] = None,
) -> Path:
    """Return the soft-label table for neural-network training."""
    return _require_file(
        _atlas_repo(atlas_repo) / "multi-atlas_soft_labels.csv",
        "Multi-atlas soft labels",
    )


def fetch_fine_grained_geodesic_distances(
    atlas_repo: Optional[PathLike] = None,
) -> Path:
    """Return the Fine-grained parcel-wise geodesic distance matrix."""
    return _require_file(
        _atlas_repo(atlas_repo)
        / "Fine-grained"
        / "geodist"
        / "parcel-wise_geodist.csv",
        "Fine-grained geodesic distance matrix",
    )


def fetch_feature_names() -> Path:
    """Return the canonical ordered feature-name table."""
    return _fetch_feature_resource("all_feature_names.csv")


__all__ = [
    "fetch_atlas",
    "fetch_fine_grained_geodesic_distances",
    "fetch_fine_grained_surface",
    "fetch_fine_grained_volume",
    "fetch_feature_names",
    "fetch_multi_atlas_label_mapping",
    "fetch_multi_atlas_soft_labels",
]
