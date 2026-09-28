"""Shared harmonization, classification, and plotting utilities."""

import colorsys
import pickle
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analysis.clinical.neuroCombat import neuroCombat


def harmonize_with_combat(
    features,
    covariates,
    batch_col,
    discrete_cols=None,
    continuous_cols=None,
):
    """Harmonize a feature table while preserving its index and columns."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not isinstance(covariates, pd.DataFrame):
        raise TypeError("covariates must be a pandas DataFrame")
    if features.empty:
        raise ValueError("features must contain at least one sample and feature")
    if not features.index.equals(covariates.index):
        raise ValueError(
            "features and covariates must have identical indexes in the same order"
        )

    discrete_cols = list(discrete_cols or [])
    continuous_cols = list(continuous_cols or [])
    required_columns = [batch_col] + discrete_cols + continuous_cols
    missing_columns = [
        column for column in required_columns if column not in covariates.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing ComBat covariates: " + ", ".join(missing_columns)
        )
    if covariates.loc[:, required_columns].isna().any().any():
        raise ValueError("ComBat covariates contain missing values")

    # ComBat is undefined for constant features. Preserve them unchanged and
    # harmonize only features that vary across the combined sample.
    variable_columns = features.columns[features.nunique(dropna=False) > 1]
    harmonized = features.astype(float).copy()
    if len(variable_columns) == 0:
        return harmonized

    harmonized_values = neuroCombat(
        data=features.loc[:, variable_columns].to_numpy(),
        covars=covariates,
        batch_col=batch_col,
        discrete_cols=discrete_cols,
        continuous_cols=continuous_cols,
    )
    harmonized.loc[:, variable_columns] = harmonized_values
    return harmonized


def _balance_binary_training_data(X, y, random_state=42):
    """Undersample the larger class within one training fold."""
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Binary classification requires exactly two classes")

    class_indices = [np.flatnonzero(y == label) for label in classes]
    n_target = min(len(indices) for indices in class_indices)
    selected = np.concatenate(
        [rng.choice(indices, n_target, replace=False) for indices in class_indices]
    )
    rng.shuffle(selected)
    return X[selected], y[selected]


def _build_svm_classifier(random_state=42):
    """Build the original nested RBF-SVM model and C-parameter search."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid={"svm__C": [0.1, 1, 5, 10]},
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
    )


def _regress_out_confounds(X_train, C_train, X_test, C_test):
    """Fit confound effects on a training fold and apply them to its test fold."""
    if C_train is None or C_test is None or C_train.shape[1] == 0:
        return X_train, X_test

    model = LinearRegression(fit_intercept=True)
    model.fit(C_train, X_train)
    return (
        X_train - model.predict(C_train),
        X_test - model.predict(C_test),
    )


def _select_l1_features(X_train, y_train, X_test, C_value=100):
    """Select nonzero L1-logistic features using only a training fold."""
    selector = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=C_value,
        random_state=42,
        max_iter=1000,
    )
    selector.fit(X_train, y_train)
    weights = np.abs(selector.coef_[0])
    selected = np.flatnonzero(weights > 0)
    if len(selected) == 0:
        selected = np.argsort(weights)[-1:]
    return X_train[:, selected], X_test[:, selected]


def _process_classification_fold(
    train_index,
    test_index,
    X,
    y,
    confounds,
    mean_fpr,
    random_state,
):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    if confounds is not None:
        C_train = confounds[train_index]
        C_test = confounds[test_index]
        X_train, X_test = _regress_out_confounds(
            X_train, C_train, X_test, C_test
        )

    X_train, X_test = _select_l1_features(X_train, y_train, X_test)
    X_train, y_train = _balance_binary_training_data(
        X_train, y_train, random_state=random_state
    )

    classifier = _build_svm_classifier(random_state=random_state)
    classifier.fit(X_train, y_train)
    y_probability = classifier.predict_proba(X_test)[:, 1]
    y_prediction = classifier.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_probability)
    interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
    interpolated_tpr[0] = 0.0
    return (
        y_test,
        y_prediction,
        y_probability,
        interpolated_tpr,
        auc(fpr, tpr),
        classifier.best_params_,
    )


def _permuted_auc(y_true, y_probability, random_state):
    rng = np.random.RandomState(random_state)
    return roc_auc_score(rng.permutation(y_true), y_probability)


def _validate_classification_inputs(features, confounds, name):
    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"{name} features must be a pandas DataFrame")
    if features.index.has_duplicates:
        raise ValueError(f"{name} feature index contains duplicate subjects")
    if features.isna().any().any():
        raise ValueError(f"{name} features contain missing values")
    if confounds is not None and not features.index.equals(confounds.index):
        raise ValueError(
            f"{name} features and confounds must have identical indexes"
        )


def run_svm_classification(
    control_features,
    disease_features,
    disease_name,
    control_confounds=None,
    disease_confounds=None,
    n_splits=10,
    n_permutations=1000,
    n_jobs=32,
    random_state=42,
    result_dir=None,
    overwrite=False,
):
    """Run nested-CV RBF-SVM classification and a label permutation test."""
    _validate_classification_inputs(
        control_features, control_confounds, "Control"
    )
    _validate_classification_inputs(
        disease_features, disease_confounds, disease_name
    )
    if not control_features.columns.equals(disease_features.columns):
        raise ValueError("Control and disease feature columns must match")
    if (control_confounds is None) != (disease_confounds is None):
        raise ValueError("Provide confounds for both groups or neither group")

    result_path = None
    if result_dir is not None:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{disease_name}_results.pkl"
        if result_path.exists() and not overwrite:
            with result_path.open("rb") as file:
                return pickle.load(file)

    X = np.vstack(
        [control_features.to_numpy(), disease_features.to_numpy()]
    )
    y = np.concatenate(
        [
            np.zeros(len(control_features), dtype=int),
            np.ones(len(disease_features), dtype=int),
        ]
    )

    if control_confounds is None:
        confounds = None
    else:
        combined_confounds = pd.concat(
            [control_confounds, disease_confounds], axis=0
        )
        combined_confounds = pd.get_dummies(
            combined_confounds, drop_first=True
        )
        confounds = combined_confounds.to_numpy(dtype=float)

    print(f"[{disease_name}] HC: {len(control_features)}")
    print(f"[{disease_name}] Cases: {len(disease_features)}")
    if confounds is not None:
        print(f"[{disease_name}] Regressing confounds")

    cross_validation = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    mean_fpr = np.linspace(0, 1, 100)
    splits = list(cross_validation.split(X, y))
    with Parallel(n_jobs=min(n_splits, n_jobs), backend="loky") as parallel:
        fold_results = parallel(
            delayed(_process_classification_fold)(
                train_index,
                test_index,
                X,
                y,
                confounds,
                mean_fpr,
                random_state,
            )
            for train_index, test_index in splits
        )

    all_y_test = np.concatenate([result[0] for result in fold_results])
    all_y_prediction = np.concatenate([result[1] for result in fold_results])
    all_y_probability = np.concatenate([result[2] for result in fold_results])
    tprs = np.asarray([result[3] for result in fold_results])
    fold_aucs = np.asarray([result[4] for result in fold_results])
    best_parameters = [result[5] for result in fold_results]

    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    cv_auc_mean = fold_aucs.mean()
    cv_auc_std = fold_aucs.std(ddof=1)

    confusion = confusion_matrix(all_y_test, all_y_prediction)
    tn, fp, fn, tp = confusion.ravel()
    accuracy = accuracy_score(all_y_test, all_y_prediction)
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = f1_score(all_y_test, all_y_prediction)
    global_auc = roc_auc_score(all_y_test, all_y_probability)

    print(f"[{disease_name}] Permutation test: {n_permutations} rounds")
    rng = np.random.RandomState(random_state)
    permutation_seeds = rng.randint(
        0, np.iinfo(np.int32).max, size=n_permutations
    )
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        permuted_aucs = parallel(
            delayed(_permuted_auc)(
                all_y_test, all_y_probability, int(seed)
            )
            for seed in permutation_seeds
        )
    permuted_aucs = np.asarray(permuted_aucs)
    p_value = (
        np.sum(permuted_aucs >= cv_auc_mean) + 1.0
    ) / (n_permutations + 1.0)

    result = {
        "disease_name": disease_name,
        "n_splits": n_splits,
        "n_permutations": n_permutations,
        "n_hc": len(control_features),
        "n_case": len(disease_features),
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "fold_tprs": tprs,
        "global_auc": global_auc,
        "fold_aucs": fold_aucs,
        "cv_auc_mean": cv_auc_mean,
        "cv_auc_std": cv_auc_std,
        "best_parameters": best_parameters,
        "permuted_aucs": permuted_aucs,
        "p_value": p_value,
        "cm": confusion,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "all_y_test": all_y_test,
        "all_y_prediction": all_y_prediction,
        "all_y_probability": all_y_probability,
    }

    if result_path is not None:
        with result_path.open("wb") as file:
            pickle.dump(result, file)

    print(f"[{disease_name}] Accuracy: {accuracy:.4f}")
    print(
        f"[{disease_name}] CV AUC: "
        f"{cv_auc_mean:.4f} +/- {cv_auc_std:.4f}"
    )
    print(f"[{disease_name}] Permutation p: {p_value:.4f}")
    return result


def _adjust_saturation(hex_color, factor=0.7):
    red, green, blue = mcolors.to_rgb(hex_color)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    saturation = np.clip(saturation * factor, 0, 1)
    return colorsys.hsv_to_rgb(hue, saturation, value)


def plot_svm_classification(
    result,
    darken_factor=0.6,
    roc_lw=1.5,
    roc_variability_alpha=0.1,
    observed_line_alpha=0.95,
    histogram_alpha=0.4,
    histogram_edgecolor="white",
    confusion_cmap="Blues",
    title_fontsize=16,
    label_fontsize=14,
    tick_fontsize=10,
    legend_fontsize=9,
    confusion_annotation_fontsize=12,
    figsize=(11, 3.8),
    spine_lw=0.8,
    edge_color="gray",
):
    """Plot ROC, permutation distribution, and confusion matrix panels."""
    if isinstance(result, (str, Path)):
        with Path(result).open("rb") as file:
            result = pickle.load(file)
    if not isinstance(result, dict):
        raise TypeError("result must be a result dictionary or pickle path")

    disease_name = result["disease_name"]
    mean_auc = result["cv_auc_mean"]
    std_auc = result["cv_auc_std"]
    p_value = result["p_value"]
    permuted_aucs = result["permuted_aucs"]
    base_color = {"SCZ": "#E64B35", "MDD": "#E64B35", "BD": "#E64B35"}.get(
        disease_name, "#1f77b4"
    )
    roc_color = _adjust_saturation(base_color, darken_factor)
    p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"
    legend_font = fm.FontProperties(
        family="DejaVu Sans", size=legend_fontsize
    )

    figure, axes = plt.subplots(1, 3, figsize=figsize)
    roc_axis, permutation_axis, confusion_axis = axes

    # Show variability across the fold-specific ROC curves, not a confidence interval.
    fold_tprs = result.get("tprs")
    if fold_tprs is None:
        fold_tprs = result.get("fold_tprs")
    if fold_tprs is not None:
        fold_tprs = np.asarray(fold_tprs)
        if fold_tprs.ndim != 2 or fold_tprs.shape[1] != len(result["mean_fpr"]):
            raise ValueError(
                "Fold ROC curves must be interpolated to the mean FPR grid"
            )
        tpr_std = np.nanstd(fold_tprs, axis=0)
        tpr_lower = np.clip(result["mean_tpr"] - tpr_std, 0, 1)
        tpr_upper = np.clip(result["mean_tpr"] + tpr_std, 0, 1)
        roc_axis.fill_between(
            result["mean_fpr"],
            tpr_lower,
            tpr_upper,
            color=roc_color,
            alpha=roc_variability_alpha,
            linewidth=0,
            zorder=1,
        )

    roc_line, = roc_axis.plot(
        result["mean_fpr"],
        result["mean_tpr"],
        color=roc_color,
        lw=roc_lw,
        alpha=0.7,
        zorder=3,
    )
    roc_axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=1.0,
        color=edge_color,
        alpha=0.8,
        zorder=2,
    )

    invisible = Patch(color="none")
    roc_legend = roc_axis.legend(
        handles=[roc_line, invisible],
        labels=[
            f"AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f}",
            f"$\it{{P}}_{{\mathrm{{perm}}}}$ {p_text}",
        ],
        loc="lower right",
        frameon=True,
        prop=legend_font,
        handlelength=0.5,
        handletextpad=0.5,
        borderpad=0.4,
        labelspacing=0.3,
    )
    roc_legend.get_frame().set_edgecolor(edge_color)
    roc_legend.get_frame().set_linewidth(0.5)
    roc_axis.set_xlim([-0.02, 1.02])
    roc_axis.set_ylim([-0.02, 1.02])
    roc_axis.set_xlabel(
        "False positive rate", fontsize=label_fontsize, labelpad=8
    )
    roc_axis.set_ylabel(
        "True positive rate", fontsize=label_fontsize, labelpad=9
    )
    roc_axis.set_title(
        f"{disease_name}  vs  HC", fontsize=title_fontsize, pad=16
    )
    roc_axis.tick_params(axis="both", labelsize=tick_fontsize)
    roc_axis.grid(False)
    for spine in roc_axis.spines.values():
        spine.set_visible(True)
        spine.set_color(edge_color)
        spine.set_linewidth(spine_lw)

    sns.histplot(
        permuted_aucs,
        bins=30,
        color="gray",
        alpha=histogram_alpha,
        edgecolor=histogram_edgecolor,
        linewidth=1.0,
        stat="density",
        ax=permutation_axis,
    )
    permutation_axis.axvline(
        x=mean_auc,
        color=roc_color,
        linestyle="--",
        linewidth=1.5,
        alpha=observed_line_alpha,
    )
    permutation_axis.legend(
        handles=[invisible, invisible],
        labels=[
            f"Observed AUC = {mean_auc:.3f}",
            f"$\it{{P}}_{{\mathrm{{perm}}}}$ {p_text}",
        ],
        loc="upper right",
        bbox_to_anchor=(0.92, 0.95),
        frameon=False,
        prop=legend_font,
        handlelength=0,
        handletextpad=0,
        borderpad=0.3,
        labelspacing=0.15,
    )
    permutation_axis.set_xlabel(
        "AUC", fontsize=label_fontsize, labelpad=8
    )
    permutation_axis.set_ylabel(
        "Density", fontsize=label_fontsize, labelpad=9
    )
    permutation_axis.set_title(
        f"Permutation test (n = {len(permuted_aucs)})",
        fontsize=title_fontsize,
        pad=16,
    )
    permutation_axis.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        direction="out",
        length=4,
        width=spine_lw,
    )
    permutation_axis.grid(False)
    permutation_axis.spines["top"].set_visible(False)
    permutation_axis.spines["right"].set_visible(False)
    for position in ("left", "bottom"):
        permutation_axis.spines[position].set_color(edge_color)
        permutation_axis.spines[position].set_linewidth(spine_lw)

    sns.heatmap(
        result["cm"],
        annot=True,
        fmt="d",
        cmap=confusion_cmap,
        xticklabels=["HC", disease_name],
        yticklabels=["HC", disease_name],
        cbar=False,
        ax=confusion_axis,
        annot_kws={"fontsize": confusion_annotation_fontsize},
        linewidths=0.5,
        linecolor=edge_color,
    )
    confusion_axis.set_xlabel(
        "Predicted label", fontsize=label_fontsize, labelpad=8
    )
    confusion_axis.set_ylabel(
        "True label", fontsize=label_fontsize, labelpad=9
    )
    confusion_axis.set_title(
        "Confusion matrix", fontsize=title_fontsize, pad=16
    )
    confusion_axis.tick_params(axis="both", labelsize=tick_fontsize)
    for spine in confusion_axis.spines.values():
        spine.set_visible(True)
        spine.set_color(edge_color)
        spine.set_linewidth(spine_lw)

    figure.tight_layout()
    plt.show()
    plt.close(figure)
    return None
