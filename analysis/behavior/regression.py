"""Reusable helpers for HCP behavioral prediction."""

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler


# Candidate ridge penalties evaluated by the inner CV loop.
alphas = [
    0.00001, 0.0001, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07,
    0.1, 0.4, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 15, 20,
]


def regress_confounds_cv(y_train, y_test, confounds_train, confounds_test):
    """Regress age/sex covariates using only the current outer-fold training set."""
    model = LinearRegression()
    model.fit(confounds_train, y_train)
    return (
        y_train - model.predict(confounds_train),
        y_test - model.predict(confounds_test),
    )


def regression_nested_cv(
    x,
    y,
    confounds,
    cv_folds=10,
    inner_cv_folds=5,
):
    """Run nested ridge CV and return correlation, predictions, and alphas.

    No random state is supplied: every call draws fresh shuffled outer and inner
    folds from the process random source.
    """
    outer_cv = KFold(n_splits=cv_folds, shuffle=True)
    y_pred_oof = np.zeros(len(y), dtype=float)
    y_true_oof = np.zeros(len(y), dtype=float)
    best_alphas = []

    for train_idx, test_idx in outer_cv.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        confounds_train = confounds[train_idx]
        confounds_test = confounds[test_idx]

        # Fit nuisance regression and scaling on outer-training subjects only.
        y_train, y_test = regress_confounds_cv(
            y_train, y_test, confounds_train, confounds_test
        )

        x_scaler = StandardScaler().fit(x_train)
        x_train = x_scaler.transform(x_train)
        x_test = x_scaler.transform(x_test)

        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True)
        search = GridSearchCV(
            Ridge(),
            param_grid={'alpha': alphas},
            scoring='r2',
            cv=inner_cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train)

        y_pred_oof[test_idx] = search.predict(x_test)
        y_true_oof[test_idx] = y_test
        best_alphas.append(search.best_params_['alpha'])

    correlation = pearsonr(y_true_oof, y_pred_oof)[0]
    return correlation, y_pred_oof, y_true_oof, best_alphas


def permutation_test(y_true, y_pred, observed_corr, n_permutations=5000):
    # default_rng() uses fresh operating-system entropy when no seed is given.
    rng = np.random.default_rng()
    perm_corrs = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        perm_corrs[index] = pearsonr(rng.permutation(y_true), y_pred)[0]
    p_value = (np.count_nonzero(perm_corrs >= observed_corr) + 1) / (n_permutations + 1)
    return perm_corrs, p_value


__all__ = [
    'alphas',
    'regress_confounds_cv',
    'regression_nested_cv',
    'permutation_test',
]
