"""Parcel-wise cortical surface visualization."""

from functools import lru_cache
import tempfile

import numpy as np
from nilearn.surface import load_surf_data
from PIL import Image

from .fetch import fetch_fine_grained_surface


@lru_cache(maxsize=1)
def _load_fine_grained_labels():
    left_path, right_path = fetch_fine_grained_surface(
        space="fs_LR", density="32k"
    )
    left_labels = np.asarray(load_surf_data(str(left_path)), dtype=int)
    right_labels = np.asarray(load_surf_data(str(right_path)), dtype=int)
    return left_labels, right_labels


def _reallocate_to_vertex(parcel_data):
    """Map parcel values to fs_LR 32k vertices using 1-based atlas labels."""
    values = np.asarray(parcel_data, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError("parcel_data must be a one-dimensional array")

    left_labels, right_labels = _load_fine_grained_labels()
    max_label = max(int(left_labels.max()), int(right_labels.max()))
    if len(values) != max_label:
        raise ValueError(
            "Expected one value for each of {} parcels, got {}".format(
                max_label, len(values)
            )
        )

    left_vertex = np.full(left_labels.shape, np.nan, dtype=float)
    right_vertex = np.full(right_labels.shape, np.nan, dtype=float)
    left_mask = left_labels > 0
    right_mask = right_labels > 0
    left_vertex[left_mask] = values[left_labels[left_mask] - 1]
    right_vertex[right_mask] = values[right_labels[right_mask] - 1]
    return left_vertex, right_vertex


def visualize_parcel_surface_map(
    parcel_data,
    title,
    cmap="YlGnBu",
    max_value=None,
    display_width=1000,
):
    """Render 7,340 Fine-grained parcel values as a cortical surface image.

    Args:
        parcel_data: One scalar per parcel in atlas label order.
        title: Title shown above the cortical surface.
        cmap: Matplotlib colormap name or colormap object.
        max_value: Optional absolute color limit. The maximum absolute finite
            parcel value is used when omitted.
        display_width: Width in pixels of the returned image.

    Returns:
        A resized ``PIL.Image.Image`` suitable for direct notebook display.
    """
    from .vis import visualize_surface_32k_fs_LR

    values = np.asarray(parcel_data, dtype=float).squeeze()
    left_vertex, right_vertex = _reallocate_to_vertex(values)

    if max_value is None:
        max_value = float(np.nanmax(np.abs(values)))
    else:
        max_value = float(max_value)
    if not np.isfinite(max_value) or max_value <= 0:
        raise ValueError("max_value must be a positive finite number")
    if not isinstance(display_width, int) or display_width <= 0:
        raise ValueError("display_width must be a positive integer")

    with tempfile.TemporaryDirectory(prefix="rise_surface_") as temp_dir:
        rendered = visualize_surface_32k_fs_LR(
            save_path=temp_dir,
            name=title,
            dpi=200,
            mymap=cmap,
            dataL=left_vertex,
            dataR=right_vertex,
            vmax=max_value,
            threshold=0,
            darkness=0.6,
            boundary=False,
            Sym=True,
        )

    display_height = round(rendered.height * display_width / rendered.width)
    resampling = getattr(Image, "Resampling", Image)
    return rendered.resize(
        (display_width, display_height),
        resampling.LANCZOS,
    )


__all__ = ["visualize_parcel_surface_map"]
