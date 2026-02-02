import numpy as np
import pytest
from dgp import SimulatedDataset


@pytest.fixture
def dataset():
    return SimulatedDataset(N=2000, d=10, alpha=1.0, seed=42)

def test_shapes(dataset):
    assert dataset.X.shape == (dataset.N, dataset.d)
    assert dataset.mu0.shape == (dataset.N,)
    assert dataset.mu1.shape == (dataset.N,)
    assert dataset.tau.shape == (dataset.N,)
    assert dataset.e.shape == (dataset.N,)
    assert dataset.W.shape == (dataset.N,)
    assert dataset.Y.shape == (dataset.N,)


def test_correlation_matrix_valid():
    d = SimulatedDataset(100, 10, alpha=1.0, seed=0)
    corr = d.generate_random_correlation_matrix()

    # symmetric
    assert np.allclose(corr, corr.T, atol=1e-8)

    # unit diagonal
    assert np.allclose(np.diag(corr), np.ones(d.d))

    # positive semidefinite
    eigvals = np.linalg.eigvalsh(corr)
    assert np.all(eigvals > -1e-8)


def test_mu0_independent_of_effect_modifiers(dataset):
    X_copy = dataset.X.copy()

    # perturb effect modifiers only
    X_copy[:, dataset.x_effect_mod_idx] += 10.0

    dataset_perturbed = SimulatedDataset(
        dataset.N, dataset.d, dataset.alpha, seed=42
    )
    dataset_perturbed.X = X_copy
    mu0_new = dataset_perturbed.set_mu0()

    assert np.allclose(dataset.mu0, mu0_new)


def test_tau_independent_of_confounders_and_prognostic(dataset):
    X_copy = dataset.X.copy()

    # perturb confounders + prognostic
    idx = dataset.x_confounders_idx + dataset.x_prognostic_idx
    X_copy[:, idx] += 10.0

    dataset_perturbed = SimulatedDataset(
        dataset.N, dataset.d, dataset.alpha, seed=42
    )
    dataset_perturbed.X = X_copy
    tau_new = dataset_perturbed.set_tau()

    assert np.allclose(dataset.tau, tau_new)


def test_mu1_definition(dataset):
    assert np.allclose(dataset.mu1, dataset.mu0 + dataset.tau)


def test_propensity_range(dataset):
    assert np.all(dataset.e > 0)
    assert np.all(dataset.e < 1)

    # basic overlap check
    assert dataset.e.min() > 0.02
    assert dataset.e.max() < 0.98

def test_propensity_independent_of_non_confounders(dataset):
    X_copy = dataset.X.copy()

    # perturb non-confounders
    non_conf = dataset.x_prognostic_idx + dataset.x_effect_mod_idx
    X_copy[:, non_conf] += 10.0

    dataset_perturbed = SimulatedDataset(
        dataset.N, dataset.d, dataset.alpha, seed=42
    )
    dataset_perturbed.X = X_copy
    e_new = dataset_perturbed.set_propensity_fn()

    assert np.allclose(dataset.e, e_new)


def test_treatment_is_binary(dataset):
    assert set(np.unique(dataset.W)).issubset({0, 1})

def test_treatment_rate_reasonable(dataset):
    rate = dataset.W.mean()
    assert 0.1 < rate < 0.9

def test_outcome_consistency(dataset):
    Y_expected = np.where(dataset.W == 1, dataset.Y1, dataset.Y0)
    assert np.allclose(dataset.Y, Y_expected)


def test_shared_noise(dataset):
    diff = dataset.Y1 - dataset.Y0
    assert np.allclose(diff, dataset.tau)
