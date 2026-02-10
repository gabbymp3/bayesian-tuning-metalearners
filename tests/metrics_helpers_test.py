import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from src.metrics_helpers import outcome_mse, pehe, cross_predict_tau
from src.xlearner import XlearnerWrapper
from src.dgp import SimulatedDataset


@pytest.fixture
def dataset():
    return SimulatedDataset(N=2000, d=10, alpha=0.5, seed=42)

def test_outcome_mse(dataset):
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(n_estimators=100, random_state=0),
        propensity_model=RandomForestClassifier(n_estimators=100, random_state=0),
        cate_models=RandomForestRegressor(n_estimators=100, random_state=0),
    )
    #wrapper.fit(dataset.Y, dataset.W, X=dataset.X)
    wrapper.fit(dataset.X, dataset.Y, W=dataset.W)
    y_pred = wrapper.predict_outcome(dataset.X, dataset.W)
    score = outcome_mse(wrapper, dataset.X, dataset.Y, dataset.W)
    assert score == np.mean((dataset.Y - y_pred) ** 2)
    assert score > 0

    # check that mse is lower when y_pred is closer to y_true
    y_pred_noisier = y_pred + np.random.normal(0, 1, size=y_pred.shape)
    score_noisier = outcome_mse(wrapper, dataset.X, y_pred_noisier, dataset.W)
    assert score_noisier > score



def test_pehe(dataset):
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(n_estimators=100, random_state=0),
        propensity_model=RandomForestClassifier(n_estimators=100, random_state=0),
        cate_models=RandomForestRegressor(n_estimators=100, random_state=0),
    )
    #wrapper.fit(dataset.Y, dataset.W, X=dataset.X)
    wrapper.fit(dataset.X, dataset.Y, W=dataset.W)
    tau_pred = wrapper.predict(dataset.X)
    score = pehe(dataset.tau, tau_pred)
    assert score == np.mean((dataset.tau - tau_pred) ** 2)
    assert score > 0

    # check that pehe is lower when tau_pred is closer to tau_true
    tau_pred_noisier = tau_pred + np.random.normal(0, 1, size=tau_pred.shape)
    score_noisier = pehe(dataset.tau, tau_pred_noisier)
    assert score_noisier > score



def test_cross_predict_tau(dataset):
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(n_estimators=100, random_state=0),
        propensity_model=RandomForestClassifier(n_estimators=100, random_state=0),
        # cate_models defaults to None (uses models)
    )
    est_class = type(wrapper)
    est_params = wrapper.get_params()
    tau_pred = cross_predict_tau(est_class, est_params, dataset.X, dataset.Y, W=dataset.W)
    assert tau_pred.shape == (dataset.N,)
    assert np.mean(tau_pred) > 0
