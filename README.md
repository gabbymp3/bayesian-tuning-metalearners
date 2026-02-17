# Hyperparameter Tuning in Causal ML Models: The Bayesian vs. Frequentist Approach

This repository contains the programmatic implementation of my (in progress) undergraduate thesis research for MMSS at Northwestern University. The objective of this research is to compare the performance of Bayesian and frequentist (grid and random) hyperparameter tuning methods for X-learner models (from `EconML`) in the context of heterogeneous treatment effect estimation.

## Objective
Causal ML combines machine learning algorithms with econometric identification strategies to estimate treatment effects; heterogeneous treatment effect estimation is particularly valuable in observational settings where the efficacy
of intervention regimes benefits from personalization. Despite promising advancements in causal ML, empirical applications remain limited by challenges that arise from both the causal side and the algorithmic sides, especially the
question of model configuration and tuning. In particular, hyperparameter tuning is frequently ad hoc or neglected in causal applications, undermining model performance and broader empirical claims on treatment effects.


This thesis seeks to address the issue of hyperparameter tuning in the development of causal ML models, asking whether a Bayesian approach to hyperparameter tuning can deliver better performance than the standard frequentist-style automatic tuning methods. The analysis focuses on causal ML methods for estimating CATE, evaluating how such tuning strategies affect the credibility of estimated treatment effects, conceptualizing the problem of tuning from two opposing statistical philosophies.

## Repository overview
`src/`
  `dgp.py` contains the `SimulatedDataset` class and the data generating function `simulate_dataset()` used in these simulations. The DGP is based on the procedure in Künzel et al. (2019) with the following modifications:

  - Confounding, prognostic, and effect modifier covariates are specified.
  - Correlation matrix is constructed from a randomly generated eigenvector.
  - Individual response functions (and thus the true treatment effect) and propensity score function are original.
  - Observed outcomes `Y0`, `Y1` constructed from their respective response functions `mu0`, `mu1` but share the same normally-distributed error term to control noise.

  `xlearner.py` contains the implentation of the X-learner model used in this study, the `XLearnerWrapper` class. This object inherits `BaseEstimator` from `sklearn.base` and wraps the XLearner object from `EconML`'s `metalearners` library.

  `metrics_helpers.py` contains the helper functions used in cross-validation and model evaluation, calculating observed outcome MSE, PEHE/PEHEplug, and TAUplug.

  `tuning.py` contains all tuning implementations, `grid_search()`, `random_search()`, and `bayesian_search()`. All tuning functions use the same internal cross-validation process, differing only in their search algorithms. Each returns the fitted model, parameters, and best score achieved after tuning.

  `experiment.py`

  `main.py`

`experiment_configs/`

**Notes**
(This readme will be updated with more information about simulation + experiment setup --  just some working notes)
- Random & Bayesian Search:
  - `n_iter` set to 10 * d for d-dimensional search space
- Base learners:
  - `catboost_info` generation
- XLearner
  - `models` vs `propensity_model` vs `cate_models`

