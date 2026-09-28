"""Surface visualization tools and packaged fs_LR 32k resources."""


def visualize_surface_32k_fs_LR(*args, **kwargs):
    """Load and call the fs_LR 32k surface renderer."""
    from .surf import visualize_surface_32k_fs_LR as render

    return render(*args, **kwargs)


__all__ = ["visualize_surface_32k_fs_LR"]
