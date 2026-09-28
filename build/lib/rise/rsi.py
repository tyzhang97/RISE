"""Regional identity signature (RSI) calculation from parcel embeddings."""

from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


PathLike = Union[str, Path]


@contextmanager
def _joblib_progress(total, description):
    """Update tqdm when joblib batches finish rather than when they are queued."""

    progress = tqdm(total=total, desc=description)
    original_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(original_callback):
        def __call__(self, *args, **kwargs):
            progress.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        progress.close()


def _repeat_directories(model_dir, repeat_names=None):
    model_dir = Path(model_dir).expanduser().resolve()
    if repeat_names is None:
        repeat_dirs = [
            path
            for pattern in ("repeat_*", "Repeat_*")
            for path in model_dir.glob(pattern)
            if path.is_dir()
        ]
        repeat_dirs.sort(key=lambda path: int(path.name.rsplit("_", 1)[1]))
    else:
        repeat_dirs = [model_dir / name for name in repeat_names]
    if not repeat_dirs:
        raise FileNotFoundError("No repeat directories found under {}".format(model_dir))
    return repeat_dirs


def _embedding_path(repeat_dir, run_name, split, hemi):
    path = repeat_dir / "results" / "run_{}_{}_embeddings_{}.h5".format(
        run_name.lower(), split.lower(), hemi
    )
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def _embedding_columns(frame):
    columns = [column for column in frame.columns if str(column).startswith("dim_")]
    columns.sort(key=lambda name: int(str(name).rsplit("_", 1)[1]))
    if not columns:
        raise ValueError("No dim_* embedding columns were found")
    return columns


def _frame_to_subject_embeddings(frame, subject_ids=None):
    """Convert one embedding table to the subject list used by the old notebook."""

    frame = frame.copy()
    frame["Subject_ID"] = frame["Subject_ID"].astype(str)
    available_subject_ids = frame["Subject_ID"].drop_duplicates().tolist()
    if subject_ids is None:
        subject_ids = available_subject_ids
    else:
        subject_ids = [str(subject_id) for subject_id in subject_ids]
        missing = sorted(set(subject_ids) - set(available_subject_ids))
        if missing:
            raise ValueError(
                "Embedding file is missing subject IDs: {}".format(
                    ", ".join(missing[:10])
                )
            )
    dim_columns = _embedding_columns(frame)
    embeddings = []
    for subject_id in subject_ids:
        subject_frame = frame.loc[frame["Subject_ID"] == subject_id, dim_columns]
        embeddings.append(subject_frame.to_numpy())
    return subject_ids, np.asarray(embeddings)


def load_split_embeddings(
    model_dir: PathLike,
    run_name: str = "mean",
    split: str = "train",
    repeat_names: Optional[Sequence[str]] = None,
    subject_ids: Optional[Sequence[str]] = None,
):
    """Load train or test embeddings from all repeated models.

    Parameters
    ----------
    model_dir : str or pathlib.Path
        Model directory containing ``repeat_*`` subdirectories.
    run_name : str, default="mean"
        Run label used in embedding filenames.
    split : {"train", "test"}, default="train"
        Dataset split to load.
    repeat_names : sequence of str, optional
        Repeat directory names to load. By default, all repeat directories are
        discovered and sorted by their numeric suffix.
    subject_ids : sequence of str, optional
        Subjects to load and their required output order. By default, subjects
        are read in the order stored in the HDF5 files.

    Returns
    -------
    loaded_subject_ids : list of str
        Subject IDs in the same order as the first dimension of every array.
    all_sample_dict_list : list of dict
        One dictionary per repeat. Its ``left`` and ``right`` arrays have shape
        ``(subjects, parcels_per_hemisphere, embedding_dim)``.
    repeat_names : tuple of str
        Repeat directory names in loading order.
    """

    repeat_dirs = _repeat_directories(model_dir, repeat_names)
    all_sample_dict_list = []
    reference_subject_ids = None

    print("Loading {} embeddings from {} repeats...".format(split, len(repeat_dirs)))
    for repeat_index, repeat_dir in enumerate(repeat_dirs):
        left_frame = pd.read_hdf(
            _embedding_path(repeat_dir, run_name, split, "lh"), key="data", mode="r"
        )
        right_frame = pd.read_hdf(
            _embedding_path(repeat_dir, run_name, split, "rh"), key="data", mode="r"
        )
        left_subject_ids, left_embeddings = _frame_to_subject_embeddings(
            left_frame, subject_ids=subject_ids
        )
        right_subject_ids, right_embeddings = _frame_to_subject_embeddings(
            right_frame, subject_ids=subject_ids
        )

        if left_subject_ids != right_subject_ids:
            raise ValueError("Left and right embedding files contain different subjects")
        if left_embeddings.shape != right_embeddings.shape:
            raise ValueError("Left and right embedding arrays have different shapes")
        if reference_subject_ids is None:
            reference_subject_ids = left_subject_ids
        elif reference_subject_ids != left_subject_ids:
            raise ValueError("Embedding subjects differ across repeats")

        all_sample_dict_list.append(
            {"left": left_embeddings, "right": right_embeddings}
        )
        print(
            "{} embeddings repeat {}/{} loaded".format(
                split, repeat_index + 1, len(repeat_dirs)
            )
        )

    print("{} embeddings loaded".format(split))
    return (
        reference_subject_ids,
        all_sample_dict_list,
        tuple(path.name for path in repeat_dirs),
    )


def build_group_reference_embeddings(train_data_all_sample_dict_list):
    """Average training embeddings separately by repeat and hemisphere.

    Parameters
    ----------
    train_data_all_sample_dict_list : list of dict
        Training embeddings returned by :func:`load_split_embeddings`.

    Returns
    -------
    train_group_mean_lh_list : list of numpy.ndarray
        Left-hemisphere group-reference embeddings. Each array has shape
        ``(parcels_per_hemisphere, embedding_dim)``.
    train_group_mean_rh_list : list of numpy.ndarray
        Right-hemisphere group-reference embeddings with the same shape.
    """

    train_group_mean_lh_list = []
    train_group_mean_rh_list = []
    repeat_count = len(train_data_all_sample_dict_list)
    print("Calculating group-reference embeddings from {} repeats...".format(repeat_count))
    for repeat_index, repeat_data in enumerate(train_data_all_sample_dict_list):
        train_group_mean_lh_list.append(repeat_data["left"].mean(axis=0))
        train_group_mean_rh_list.append(repeat_data["right"].mean(axis=0))
        print(
            "Group-reference embeddings repeat {}/{} calculated".format(
                repeat_index + 1, repeat_count
            )
        )
    print("Group-reference embeddings calculated")
    return train_group_mean_lh_list, train_group_mean_rh_list


def compute_similarity(A, B, method="pearson"):
    """Calculate parcel-by-parcel similarity using the historical formulas.

    Parameters
    ----------
    A : numpy.ndarray
        Group-reference embeddings with shape ``(reference_parcels, dimensions)``.
    B : numpy.ndarray
        Individual embeddings with shape ``(individual_parcels, dimensions)``.
    method : {"pearson", "cosine", "euclidean"}, default="pearson"
        Similarity metric. Euclidean distances are converted to row-wise
        similarities with ``1 - distance / row_max``.

    Returns
    -------
    numpy.ndarray
        Similarity matrix with shape
        ``(reference_parcels, individual_parcels)``.
    """

    A = np.asarray(A).astype(np.float32)
    B = np.asarray(B).astype(np.float32)

    if method == "pearson":
        A_centered = A - A.mean(axis=1, keepdims=True)
        B_centered = B - B.mean(axis=1, keepdims=True)
        ss_A = np.sum(A_centered ** 2, axis=1)
        ss_B = np.sum(B_centered ** 2, axis=1)
        return np.dot(A_centered, B_centered.T) / np.sqrt(
            np.dot(ss_A[:, None], ss_B[None])
        )

    if method == "cosine":
        return cosine_similarity(A, B)

    if method == "euclidean":
        A_square = np.sum(A ** 2, axis=1).reshape(-1, 1)
        B_square = np.sum(B ** 2, axis=1).reshape(1, -1)
        distances = np.sqrt(np.maximum(A_square - 2 * (A @ B.T) + B_square, 0.0))
        row_max = distances.max(axis=1, keepdims=True)
        return np.divide(
            row_max - distances,
            row_max,
            out=np.zeros_like(distances),
            where=row_max > 0,
        )

    raise ValueError("method must be 'pearson', 'cosine', or 'euclidean'")


def _compute_single_subject_expression(
    subject_index,
    out_array,
    repeat_num,
    train_group_mean_lh_list,
    train_group_mean_rh_list,
    data_all_sample_dict_list,
    method,
    hemisphere_size,
):
    """Old notebook worker, storing a compact matrix instead of zero blocks."""

    lh_expression_matrix_list = []
    rh_expression_matrix_list = []
    for repeat_index in range(repeat_num):
        lh_expression_matrix_list.append(
            compute_similarity(
                train_group_mean_lh_list[repeat_index],
                data_all_sample_dict_list[repeat_index]["left"][subject_index],
                method=method,
            )
        )
        rh_expression_matrix_list.append(
            compute_similarity(
                train_group_mean_rh_list[repeat_index],
                data_all_sample_dict_list[repeat_index]["right"][subject_index],
                method=method,
            )
        )

    out_array[subject_index, :hemisphere_size, :] = np.mean(
        np.asarray(lh_expression_matrix_list), axis=0
    )
    out_array[subject_index, hemisphere_size:, :] = np.mean(
        np.asarray(rh_expression_matrix_list), axis=0
    )


def compute_region_corrs_parallel(
    subject_ids,
    train_group_mean_lh_list,
    train_group_mean_rh_list,
    data_all_sample_dict_list,
    repeat_num=None,
    similarity="cosine",
    n_jobs=16,
):
    """Calculate all individual expression matrices in parallel.

    Parameters
    ----------
    subject_ids : sequence of str
        Subject IDs defining the first dimension and its order.
    train_group_mean_lh_list, train_group_mean_rh_list : list of numpy.ndarray
        Repeat-specific group-reference embeddings returned by
        :func:`build_group_reference_embeddings`.
    data_all_sample_dict_list : list of dict
        Train or test embeddings returned by :func:`load_split_embeddings`.
    repeat_num : int, optional
        Number of repeated models to average. The default uses every repeat.
    similarity : {"pearson", "cosine", "euclidean"}, default="cosine"
        Parcel similarity metric.
    n_jobs : int, default=16
        Number of joblib threads used across subjects.

    Returns
    -------
    numpy.ndarray
        Individual expression matrices with shape
        ``(subjects, 2 * parcels_per_hemisphere, parcels_per_hemisphere)``.
        For the Fine-grained atlas this is ``(subjects, 7340, 3670)``. Left and
        right matrices are concatenated along rows; cross-hemisphere zero blocks
        are not stored.
    """

    if repeat_num is None:
        repeat_num = len(data_all_sample_dict_list)
    if not (
        repeat_num
        == len(train_group_mean_lh_list)
        == len(train_group_mean_rh_list)
        == len(data_all_sample_dict_list)
    ):
        raise ValueError("repeat_num does not match the embedding lists")

    hemisphere_size = train_group_mean_lh_list[0].shape[0]
    indiv_expression_matrices = np.zeros(
        (len(subject_ids), hemisphere_size * 2, hemisphere_size)
    )
    print(
        "Calculating individual expression matrices for {} subjects...".format(
            len(subject_ids)
        )
    )
    with _joblib_progress(len(subject_ids), "Individual expression matrices"):
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_single_subject_expression)(
                subject_index,
                indiv_expression_matrices,
                repeat_num,
                train_group_mean_lh_list,
                train_group_mean_rh_list,
                data_all_sample_dict_list,
                similarity,
                hemisphere_size,
            )
            for subject_index in range(len(subject_ids))
        )
    print("Individual expression matrices calculated")
    return indiv_expression_matrices


def build_group_expression_matrix(train_indiv_expression_matrices):
    """Restore the training group mean to the historical square matrix.

    Parameters
    ----------
    train_indiv_expression_matrices : numpy.ndarray
        Compressed training matrices returned by
        :func:`compute_region_corrs_parallel`, with shape
        ``(subjects, 2 * parcels_per_hemisphere, parcels_per_hemisphere)``.

    Returns
    -------
    numpy.ndarray
        Block-diagonal group mean with shape
        ``(2 * parcels_per_hemisphere, 2 * parcels_per_hemisphere)``. For the
        Fine-grained atlas this is ``(7340, 7340)``. Cross-hemisphere blocks are
        zero, matching the original RSI threshold calculation.
    """

    compressed_group_mean = np.mean(train_indiv_expression_matrices, axis=0)
    hemisphere_size = compressed_group_mean.shape[1]
    group_expression_matrix = np.zeros(
        (hemisphere_size * 2, hemisphere_size * 2)
    )
    group_expression_matrix[:hemisphere_size, :hemisphere_size] = (
        compressed_group_mean[:hemisphere_size]
    )
    group_expression_matrix[hemisphere_size:, hemisphere_size:] = (
        compressed_group_mean[hemisphere_size:]
    )
    print("Training group expression matrix calculated")
    return group_expression_matrix


def calculate_adaptive_threshold(
    training_group_expression_matrix, percentile=99.0
):
    """Calculate the adaptive RSI threshold from the training group matrix.

    Parameters
    ----------
    training_group_expression_matrix : numpy.ndarray
        Square block-diagonal training group expression matrix returned by
        :func:`build_group_expression_matrix`.
    percentile : float, default=99.0
        Row-wise percentile to calculate before averaging across parcels.

    Returns
    -------
    float
        Mean of the row-wise percentile values, rounded to two decimal places.
    """

    row_thresholds = np.percentile(
        training_group_expression_matrix, percentile, axis=1
    )
    threshold = round(float(np.mean(row_thresholds)), 2)
    print("Adaptive threshold calculated: {:.2f}".format(threshold))
    return threshold


def _compute_single_subject_rsi(subject_matrix, threshold, threshold_bins):
    single_bin_rsi = np.sum(
        np.where(subject_matrix > threshold, subject_matrix, 0.0), axis=1
    )
    num_bins = len(threshold_bins) - 1
    multi_bin_rsi = np.zeros((subject_matrix.shape[0], num_bins))
    for bin_index in range(num_bins):
        lower_bound = threshold_bins[bin_index]
        upper_bound = threshold_bins[bin_index + 1]
        if bin_index == 0:
            mask = (subject_matrix >= lower_bound) & (subject_matrix <= upper_bound)
        else:
            mask = (subject_matrix > lower_bound) & (subject_matrix <= upper_bound)
        multi_bin_rsi[:, bin_index] = np.sum(subject_matrix * mask, axis=1)
    return single_bin_rsi, multi_bin_rsi


def calculate_rsi_parallel(
    indiv_expression_matrices,
    subject_ids,
    threshold,
    num_bins=10,
    n_jobs=16,
):
    """Calculate single-bin and multi-bin RSI from cached expression matrices.

    Parameters
    ----------
    indiv_expression_matrices : numpy.ndarray
        Cached matrices with shape
        ``(subjects, 2 * parcels_per_hemisphere, parcels_per_hemisphere)``.
    subject_ids : sequence of str
        IDs corresponding to the first matrix dimension.
    threshold : float
        Adaptive threshold calculated from the training group matrix.
    num_bins : int, default=10
        Number of equally spaced bins between ``threshold`` and 1.
    n_jobs : int, default=16
        Number of joblib threads used across subjects.

    Returns
    -------
    single_bin_rsi : pandas.DataFrame
        Thresholded similarity sums with shape ``(subjects, parcels)``.
    multi_bin_rsi : numpy.ndarray
        Per-bin similarity sums with shape ``(subjects, parcels, num_bins)``.
    """

    threshold_bins = np.linspace(threshold, 1.0, num_bins + 1)
    print("Calculating RSI for {} subjects...".format(len(subject_ids)))
    with _joblib_progress(len(subject_ids), "RSI"):
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_single_subject_rsi)(
                indiv_expression_matrices[subject_index], threshold, threshold_bins
            )
            for subject_index in range(len(subject_ids))
        )
    single_bin_rsi = np.asarray([result[0] for result in results])
    multi_bin_rsi = np.asarray([result[1] for result in results])
    columns = ["r{}".format(index) for index in range(single_bin_rsi.shape[1])]
    single_bin_rsi = pd.DataFrame(
        single_bin_rsi,
        index=[str(subject_id) for subject_id in subject_ids],
        columns=columns,
    )
    single_bin_rsi.index.name = "Subject_ID"
    print("RSI calculated")
    return single_bin_rsi, multi_bin_rsi


def compute_training_rsi(
    train_indiv_expression_matrices,
    train_subject_ids,
    threshold,
    num_bins=10,
    n_jobs=16,
):
    """Calculate training RSI and its raw parcel-wise group mean.

    Parameters
    ----------
    train_indiv_expression_matrices : numpy.ndarray
        Cached training expression matrices with shape
        ``(subjects, 2 * parcels_per_hemisphere, parcels_per_hemisphere)``.
    train_subject_ids : sequence of str
        Training subject IDs in matrix order.
    threshold : float
        Adaptive threshold derived from the training group matrix.
    The function always returns raw single-bin and multi-bin RSI. Normalize
    single-bin values explicitly in the calling workflow when needed.
    num_bins : int, default=10
        Number of multi-bin RSI intervals.
    n_jobs : int, default=16
        Number of joblib threads used across subjects.

    Returns
    -------
    train_rsi : pandas.DataFrame
        Training single-bin RSI with shape ``(subjects, parcels)``. This is
        normalized only when ``norm=True``.
    train_multibin_rsi : numpy.ndarray
        Training multi-bin RSI with shape ``(subjects, parcels, num_bins)``.
    train_group_mean : pandas.Series
        Raw, unnormalized training mean single-bin RSI for every parcel. Pass
        this value to :func:`compute_test_rsi`.
    train_multibin_group_mean : numpy.ndarray
        Raw training mean multi-bin RSI with shape ``(parcels, num_bins)``.
        Pass this value to :func:`compute_test_rsi`.
    """

    train_rsi, train_multibin_rsi = calculate_rsi_parallel(
        train_indiv_expression_matrices,
        train_subject_ids,
        threshold,
        num_bins=num_bins,
        n_jobs=n_jobs,
    )
    train_group_mean = train_rsi.mean(axis=0)
    train_group_mean.name = "train_group_mean_rsi"
    train_multibin_group_mean = train_multibin_rsi.mean(axis=0)
    return (
        train_rsi,
        train_multibin_rsi,
        train_group_mean,
        train_multibin_group_mean,
    )


def compute_test_rsi(
    test_indiv_expression_matrices,
    test_subject_ids,
    threshold,
    training_group_mean,
    training_multibin_group_mean,
    num_bins=10,
    n_jobs=16,
):
    """Calculate test RSI using references and normalization from training.

    Parameters
    ----------
    test_indiv_expression_matrices : numpy.ndarray
        Cached test expression matrices with shape
        ``(subjects, 2 * parcels_per_hemisphere, parcels_per_hemisphere)``.
    test_subject_ids : sequence of str
        Test subject IDs in matrix order.
    threshold : float
        Adaptive threshold derived only from training data.
    training_group_mean : pandas.Series or numpy.ndarray
        Raw parcel-wise mean returned by :func:`compute_training_rsi`.
    training_multibin_group_mean : numpy.ndarray
        Raw parcel-wise and bin-wise mean returned by
        :func:`compute_training_rsi`, with shape ``(parcels, num_bins)``.
    The function always returns raw single-bin and multi-bin RSI. Normalize
    single-bin values explicitly in the calling workflow when needed.
    num_bins : int, default=10
        Number of multi-bin RSI intervals.
    n_jobs : int, default=16
        Number of joblib threads used across subjects.

    Returns
    -------
    test_rsi : pandas.DataFrame
        Test single-bin RSI with shape ``(subjects, parcels)``.
    test_multibin_rsi : numpy.ndarray
        Test multi-bin RSI with shape ``(subjects, parcels, num_bins)``.
    """

    test_rsi, test_multibin_rsi = calculate_rsi_parallel(
        test_indiv_expression_matrices,
        test_subject_ids,
        threshold,
        num_bins=num_bins,
        n_jobs=n_jobs,
    )
    return test_rsi, test_multibin_rsi
