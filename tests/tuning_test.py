import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from skopt.space import Integer, Categorical, Real
from scipy.stats import randint
from econml.metalearners import XLearner
from src.dgp import SimulatedDataset
from src.xlearner import XlearnerWrapper
from src.tuning import grid_search, random_search, bayesian_search


@pytest.fixture
def dataset():
    return SimulatedDataset(N=2000, d=10, alpha=0.5, seed=42)


# -----------------------------
# Single-method tests
# -----------------------------
def test_tune_grid_search(dataset):
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(random_state=0),
        propensity_model=RandomForestClassifier(random_state=0),
    )
    param_grid = {
        'models__n_estimators': [10, 20],
        'models__max_depth': [3, 5],
    }

    best_wrapper, best_params, best_score = grid_search(
        wrapper, param_grid, dataset.X, dataset.Y, dataset.W, cv=3, verbose=0
    )

    assert isinstance(best_wrapper, XlearnerWrapper)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert best_score > 0


def test_tune_random_search(dataset):
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(random_state=0),
        propensity_model=RandomForestClassifier(random_state=0),
    )

    param_dist = {
        'models__n_estimators': Integer(10, 30),
        'models__max_depth': Integer(3, 7),
    }

    best_wrapper, best_params, best_score = random_search(
        wrapper, param_dist, dataset.X, dataset.Y, dataset.W, cv=3, n_iter=5, verbose=0
    )

    assert isinstance(best_wrapper, XlearnerWrapper)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert best_score > 0


def test_tune_bayesian_search(dataset):
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(random_state=0),
        propensity_model=RandomForestClassifier(random_state=0),
    )

    param_dist = {
        'models__n_estimators': Integer(10, 30),
        'models__max_depth': Integer(3, 7),
    }

    best_wrapper, best_params, best_score = bayesian_search(
        estimator=wrapper,
        param_dist=param_dist,
        X=dataset.X,
        Y=dataset.Y,
        W=dataset.W,
        cv=3,
        n_iter=5,
        verbose=False
    )

    assert isinstance(best_wrapper, XlearnerWrapper)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert set(best_params.keys()) == set(param_dist.keys())
    assert best_score > 0


# -----------------------------
# Parameterized test for multiple search methods and base models
# -----------------------------
@pytest.mark.parametrize("base_model", ["rf"])
@pytest.mark.parametrize("search_fn", [grid_search, random_search, bayesian_search])
def test_tuning_methods(dataset, base_model, search_fn):
    if base_model == "rf":
        wrapper = XlearnerWrapper(
            models=RandomForestRegressor(random_state=0),
            propensity_model=RandomForestClassifier(random_state=0),
        )
        space_grid = {
            'models__n_estimators': [10, 20],
            'models__max_depth': [3, 5],
        }
        space_random = {
            'models__n_estimators': Integer(10, 30),
            'models__max_depth': Integer(3, 7),
        }
        space_bayes = {
            'models__n_estimators': Integer(10, 30),
            'models__max_depth': Integer(3, 7),
        }


    if search_fn is grid_search:
        param_space = space_grid
        best_wrapper, best_params, best_score = search_fn(
            wrapper,
            param_space,
            dataset.X,
            dataset.Y,
            dataset.W,
            cv=3,
            verbose=0
        )
    elif search_fn is random_search:
        param_space = space_random
        best_wrapper, best_params, best_score = search_fn(
            wrapper,
            param_space,
            dataset.X,
            dataset.Y,
            dataset.W,
            cv=3,
            n_iter=5,
            verbose=0
        )
    else:  # bayesian_search
        param_space = space_bayes
        best_wrapper, best_params, best_score = search_fn(
            estimator=wrapper,
            param_dist=param_space,
            X=dataset.X,
            Y=dataset.Y,
            W=dataset.W,
            cv=3,
            n_iter=5,
            verbose=False
        )

    assert isinstance(best_wrapper, XlearnerWrapper)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert best_score > 0