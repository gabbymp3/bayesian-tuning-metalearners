import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.metalearners import XLearner
from src.dgp import SimulatedDataset
from src.xlearner import XlearnerWrapper



@pytest.fixture
def dataset():
    return SimulatedDataset(N=2000, d=10, alpha=1.0, seed=42)

@pytest.fixture
def naive_dataset():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    W = np.array([0, 1, 0])
    Y = np.array([1, 2, 3])
    return X, W, Y

@pytest.fixture
def zero_tau_dataset():
    dgp = SimulatedDataset(N=2000, d=10, alpha=0.5, seed=42)
    dgp.tau = np.zeros(dgp.N)
    dgp.mu1 = dgp.mu0.copy()
    dgp.Y1 = dgp.Y0.copy()
    dgp.Y = dgp.Y0.copy()
    return dgp

def test_xlearner_wrapper_init():
    wrapper = XlearnerWrapper(models=LinearRegression(),
                              propensity_model=LinearRegression(),
                              cate_models=LinearRegression())
    assert wrapper.models is not None
    assert wrapper.propensity_model is not None
    assert wrapper.cate_models is not None
    assert wrapper._est is None
    #assert wrapper.true_tau is None
    assert wrapper.predict is not None
    assert wrapper.predict_outcome is not None
    assert wrapper.set_params is not None
    assert wrapper.get_params is not None

def test_xlearner_wrapper_fit(dataset):
    wrapper = XlearnerWrapper(models=LinearRegression(),
                              propensity_model=RandomForestClassifier(),
                              cate_models=LinearRegression())
    wrapper.fit(dataset.Y, dataset.W, X=dataset.X)
    assert wrapper._est is not None
    #assert wrapper.true_tau is None
    assert isinstance(wrapper._est, XLearner)
    assert wrapper.predict is not None
    assert wrapper.predict_outcome is not None
    assert wrapper.set_params is not None
    assert wrapper.get_params is not None


def test_xlearner_wrapper_predict_naive(naive_dataset):
    X, W, Y = naive_dataset
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(n_estimators=10, random_state=0),
        propensity_model=RandomForestClassifier(n_estimators=10, random_state=0),
        cate_models=RandomForestRegressor(n_estimators=10, random_state=0),
    )

    assert wrapper._est is None
    wrapper.fit(Y, W, X=X)
    assert wrapper._est is not None
    assert isinstance(wrapper._est, XLearner)

    tau_hat = wrapper.predict(X)
    assert tau_hat is not None
    assert tau_hat.shape == (X.shape[0],)

    y_hat = wrapper.predict_outcome(X, W)
    assert y_hat is not None
    assert y_hat.shape == (X.shape[0],)


def test_predict_outcome_responds_to_treatment_change(naive_dataset):
    # sanity check
    X, W, Y = naive_dataset
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(n_estimators=50, random_state=0),
        propensity_model=RandomForestClassifier(n_estimators=50, random_state=0),
        cate_models=RandomForestRegressor(n_estimators=50, random_state=0),
    )
    wrapper.fit(Y, W, X=X)

    y_hat_original = wrapper.predict_outcome(X, W)
    W_flipped = 1 - W
    y_hat_flipped = wrapper.predict_outcome(X, W_flipped)

    assert y_hat_original.shape == y_hat_flipped.shape
    assert np.any(y_hat_original != y_hat_flipped)


def test_zero_tau(zero_tau_dataset):
    wrapper = XlearnerWrapper(
        models=RandomForestRegressor(n_estimators=100, random_state=0),
        propensity_model=RandomForestClassifier(n_estimators=100, random_state=0),
        cate_models=RandomForestRegressor(n_estimators=100, random_state=0),
    )
    wrapper.fit(zero_tau_dataset.Y, zero_tau_dataset.W, X=zero_tau_dataset.X)
    tau_hat = wrapper.predict(zero_tau_dataset.X)
    assert np.allclose(tau_hat, np.zeros(zero_tau_dataset.N), atol=1.5)
