# Hyperparameter Tuning in Causal ML Models: The Bayesian vs. Frequentist Approach

This repository contains the programmatic implementation of my (in progress) undergraduate thesis research for MMSS at Northwestern University. The objective of this research is to compare the performance of Bayesian and frequentist (grid and random) hyperparameter tuning methods for X-learner models (from `EconML`) in the context of heterogeneous treatment effect estimation.

## Objective
Causal ML combines machine learning algorithms with econometric identification strategies to estimate treatment effects; heterogeneous treatment effect estimation is particularly valuable in observational settings where the efficacy of intervention regimes benefits from personalization. Despite promising advancements in causal ML, empirical applications remain limited by challenges that arise from both the causal side and the algorithmic sides, especially the question of model configuration and tuning. In particular, hyperparameter tuning is frequently ad hoc or neglected in causal applications, undermining model performance and broader empirical claims on treatment effects.


This thesis seeks to address the issue of hyperparameter tuning in the development of causal ML models, asking whether a Bayesian approach to hyperparameter tuning can deliver better performance than the standard frequentist-style automatic tuning methods. The analysis focuses on causal ML methods for estimating CATE, evaluating how such tuning strategies affect the credibility of estimated treatment effects, conceptualizing the problem of tuning from two opposing statistical philosophies.

## Repository overview

```text
bayesian-tuning-metalearners/
├── src/
│   ├── dgp.py
│   └── xlearner.py
│   └── metrics_helpers.py
│   └── tuning.py
│   └── experiment.py
│   └── main.py
│   └── experiment_configs/
│       ├── config_1d.py
│       └── config_2d.py
│       └── config_4d.py
│       └── config62d.py
├── pyproject.toml
└── README.md
```


`src/`
  `dgp.py` contains the `SimulatedDataset` class and the data generating function `simulate_dataset()` used in these simulations. The DGP is based on the procedure in Künzel et al. (2019) with the following modifications:

  - Confounding, prognostic, and effect modifier covariates are specified.
  - Correlation matrix is constructed from a randomly generated eigenvector.
  - Individual response functions (and thus the true treatment effect) and propensity score function are original.
  - Observed outcomes `Y0`, `Y1` constructed from their respective response functions `mu0`, `mu1` but share the same normally-distributed error term to control noise.

  `xlearner.py` contains the implentation of the X-learner model used in this study, the `XLearnerWrapper` class. This object inherits `BaseEstimator` from `sklearn.base` and wraps the XLearner object from `EconML`'s `metalearners` library.

  `metrics_helpers.py` contains the helper functions used in cross-validation and model evaluation, calculating observed outcome MSE, PEHE/PEHEplug, and TAUplug.

  `tuning.py` contains all tuning implementations, `grid_search()`, `random_search()`, and `bayesian_search()`. All tuning functions use the same internal cross-validation process, differing only in their search algorithms. Each returns the fitted model, parameters, and best score achieved after tuning.

  `experiment.py` implements one Monte-Carlo simulation using R repetitions. The experimental workflow is as follows: 
  - Given a data-generating function, base learner configuration, tuner configuration, and R value:
    - For each Monte-Carlo repetition 1 through R, simulate a training and test dataset using the same DGP parameters `dgp_params` and a different random seed. Then construct an `Xlearner` with parameters given by the base learner setup, `learner_config`.
      - For each tuner configuration in `tuners`, tune an XLearner model on the training data. Then, use the test data to estimate CATE `tau_hat` and cross-predict `tau_plug`, and calculate PEHE and PEHE plugin values.
      - Store the learner-tuner combination and its resulting PEHE and PEHE plugin values in `raw_results`, and generate a summary table containing the mean and variance of both PEHE metrics across Monte-Carlo simulations.

  `main.py` conducts the entire pipeline, running all specified experiments, and storing their results in an output directory with the following format:

  ```text
bayesian-tuning-metalearners/
├── results_R_{R value}/
│   ├── x_cb/
│       ├── 1d/
│         └── raw_results.csv
│         └── summary.csv
│       ├── 2d/ ...
│       ├── 4d/ ...
│       ├── 6d/ ...
│   └── x_rf/
│       ├── 1d/ ...
│       ├── 2d/ ...
│       ├── 4d/ ...
|       ├── 6d/ ...
```

  
**Notes**
- Random & Bayesian Search:
  - `n_iter` set to 10 * d for d-dimensional search space
- Base learners:
  - `catboost_info` generation
- XLearner
  - `models` vs `propensity_model` vs `cate_models`

