"""
Example: Simplified usage of XlearnerWrapper with optional cate_models

This demonstrates how to set up experiments with the simplified API.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from src.experiment import run_experiment
from src.dgp import simulate_dataset
from src.tuning import grid_search, random_search, bayesian_search
from skopt.space import Integer

# ============================================================================
# SIMPLIFIED LEARNER CONFIGURATION
# ============================================================================

learners = [
    {
        "name": "x_rf",
        "models": RandomForestRegressor(random_state=0),
        "propensity_model": RandomForestClassifier(n_estimators=50, random_state=0),
    },
    {
        "name": "x_rf_small",
        "models": RandomForestRegressor(n_estimators=20, random_state=0),
        "propensity_model": RandomForestClassifier(n_estimators=20, random_state=0),
    }
]

# ============================================================================
# TUNER CONFIGURATION
# ============================================================================

tuners = [
    {
        "name": "grid_search",
        "fn": grid_search,
        "param_space": {
            "models__n_estimators": [50, 100],
            "models__max_depth": [5, 10],
        },
        "kwargs": {"cv": 3, "verbose": False}
    },
    {
        "name": "random_search",
        "fn": random_search,
        "param_space": {
            "models__n_estimators": [50, 100, 150],
            "models__max_depth": [3, 5, 10],
            "models__min_samples_leaf": [1, 5, 10],
        },
        "kwargs": {"cv": 3, "n_iter": 10, "verbose": False}
    },
    {
        "name": "bayesian_search",
        "fn": bayesian_search,
        "param_space": {
            "models__n_estimators": Integer(50, 200),
            "models__max_depth": Integer(3, 15),
            "models__min_samples_leaf": Integer(1, 20),
        },
        "kwargs": {"cv": 3, "n_iter": 10, "verbose": False}
    }
]

# ============================================================================
# DGP CONFIGURATION
# ============================================================================

dgp_params = {
    "N": 300,      # Sample size
    "d": 10,       # Number of features
    "alpha": 0.3   # Confounding strength
}

# ============================================================================
# RUN EXPERIMENT
# ============================================================================

print("=" * 70)
print("Running Experiment with Simplified XlearnerWrapper API")
print("=" * 70)
print(f"\nLearners: {len(learners)}")
for learner in learners:
    print(f"  - {learner['name']}")
print(f"\nTuners: {len(tuners)}")
for tuner in tuners:
    print(f"  - {tuner['name']}")
print(f"\nDGP: N={dgp_params['N']}, d={dgp_params['d']}, alpha={dgp_params['alpha']}")
print(f"Repetitions: R=3")
print("\n" + "-" * 70)

summary, raw = run_experiment(
    learners=learners,
    tuners=tuners,
    R=3,
    simulate_dataset_fn=simulate_dataset,
    dgp_params=dgp_params,
    base_seed=42,
    cv_plug=3
)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for result in summary:
    print(f"\n{result['learner']} + {result['tuner']}")
    print(f"  PEHE Mean:        {result['pehe_mean']:.4f}")
    print(f"  PEHE Variance:    {result['pehe_var']:.4f}")
    print(f"  PEHE Plug Mean:   {result['pehe_plug_mean']:.4f}")
    print(f"  PEHE Plug Var:    {result['pehe_plug_var']:.4f}")

print("\n" + "=" * 70)
print("✓ Experiment completed successfully!")
print("=" * 70)

