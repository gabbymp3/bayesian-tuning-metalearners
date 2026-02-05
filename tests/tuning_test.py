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
from src.tuning import grid_search, random_search

@pytest.fixture
def dataset():
    return SimulatedDataset(N=2000, d=10, alpha=0.5, seed=42)

def test_tune_grid_search(dataset):
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(random_state=0),
        propensity_model=RandomForestClassifier(random_state=0),
        cate_models=RandomForestRegressor(random_state=0),
    )
    param_grid = {
        'models__n_estimators': [50, 100, 200],
        'models__max_depth': [3, 5, 8],
        'models__min_samples_leaf': [1, 5, 10],
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
        cate_models=RandomForestRegressor(random_state=0),
    )

    param_dist = {
        'models__n_estimators': randint(50, 200),
        'models__max_depth': randint(3, 9),
        'models__min_samples_leaf': randint(1, 11)
    }

    best_wrapper, best_params, best_score = random_search(
        wrapper, param_dist, dataset.X, dataset.Y, dataset.W, cv=3, verbose=0
    )
    assert isinstance(best_wrapper, XlearnerWrapper)
    assert isinstance(best_params, dict)
    assert isinstance(best_score, float)
    assert best_score > 0