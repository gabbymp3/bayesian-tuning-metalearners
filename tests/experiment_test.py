import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from src.experiment import run_experiment
from src.dgp import SimulatedDataset
from src.tuning import grid_search

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def learner_config():
    return {
        "name": "x_rf",
        "models": RandomForestRegressor(n_estimators=5, random_state=0),
        "propensity_model": RandomForestClassifier(n_estimators=5, random_state=0),
    }

@pytest.fixture
def tuners():
    return [
        {
            "name": "grid",
            "fn": grid_search,
            "param_grid": {"models__n_estimators": [5, 10]},
            "kwargs": {"cv": 3}
        }
    ]

@pytest.fixture
def dgp_params():
    # Set d >= 10 to match covariate indexing in SimulatedDataset
    return {"N": 100, "d": 10, "alpha": 0.5}

@pytest.fixture
def dataset_fn():
    # Wrapper to match new simulate_dataset signature
    def wrapper(dgp_params, seed=0):
        dgp = SimulatedDataset(
            N=dgp_params.get("N", 100),
            d=dgp_params.get("d", 10),
            alpha=dgp_params.get("alpha", 0.5),
            seed=seed
        )
        return dgp.X, dgp.W, dgp.Y, dgp.mu0, dgp.mu1, dgp.Y0, dgp.Y1, dgp.tau, dgp.e
    return wrapper

# -----------------------------
# Tests
# -----------------------------
@pytest.mark.parametrize("R", [1, 2])
def test_run_experiment_basic(learner_config, tuners, dgp_params, dataset_fn, R):
    """
    Run a small experiment for a single learner and check output shapes and keys.
    """

    summary, raw = run_experiment(
        learner_config=learner_config,
        tuners=tuners,
        R=R,
        simulate_dataset_fn=dataset_fn,
        dgp_params=dgp_params,
        base_seed=42,
        cv_plug=3  # use fewer folds for speed in tests
    )

    # Basic checks
    assert isinstance(summary, list)
    assert len(summary) == len(tuners)  # Only one learner

    assert isinstance(raw, dict)
    for tuner_name, val in raw.items():
        assert "pehe" in val
        assert "pehe_plug" in val
        assert val["pehe"].shape[0] == R
        assert val["pehe_plug"].shape[0] == R

    # Ensure numeric values
    for record in summary:
        assert "pehe_mean" in record
        assert isinstance(record["pehe_mean"], float)
        assert "pehe_var" in record
        assert isinstance(record["pehe_var"], float)
        assert "pehe_plug_mean" in record
        assert isinstance(record["pehe_plug_mean"], float)
        assert "pehe_plug_var" in record
        assert isinstance(record["pehe_plug_var"], float)
