import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import product
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from src.metrics_helpers import outcome_mse
from src.xlearner import XlearnerWrapper



def expand_param_grid(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def grid_search(estimator, param_grid, X, Y, W, cv=5, random_state=123, verbose=False, n_jobs=-1):
    """
    Manual GridSearchCV for XlearnerWrapper
    """

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    best_score = np.inf
    best_params = None
    best_estimator = None

    for params in expand_param_grid(param_grid):
        fold_scores = []

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            W_tr, W_te = W[train_idx], W[test_idx]

            if len(np.unique(W_te)) < 2:
                continue

            est = clone(estimator)
            est.set_params(**params)
            est.fit(X_tr, Y_tr, W=W_tr)

            mse = outcome_mse(est, X_te, Y_te, W_te)
            fold_scores.append(mse)

        avg_score = np.mean(fold_scores)

        if verbose:
            print(f"Params {params} | MSE = {avg_score:.4f}")

        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            best_estimator = clone(estimator).set_params(**params)
            best_estimator.fit(X, Y, W=W)

    return best_estimator, best_params, best_score