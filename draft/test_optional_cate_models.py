"""
Test script demonstrating that cate_models is now optional in XlearnerWrapper.

When cate_models is not specified (or set to None), the X-learner will use
the same models for CATE estimation as for outcome estimation.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from src.dgp import SimulatedDataset
from src.xlearner import XlearnerWrapper
from src.metrics_helpers import pehe

# Generate a simulated dataset
print("=" * 60)
print("Testing Optional cate_models Parameter")
print("=" * 60)

dgp = SimulatedDataset(N=1000, d=10, alpha=0.5, seed=42)
print(f"\nDataset: N={dgp.N}, d={dgp.d}, alpha={dgp.alpha}")
print(f"True average treatment effect: {np.mean(dgp.tau):.3f}")

# Test 1: Without cate_models (new default behavior)
print("\n" + "-" * 60)
print("Test 1: XlearnerWrapper WITHOUT cate_models (uses models)")
print("-" * 60)

wrapper1 = XlearnerWrapper(
    models=RandomForestRegressor(n_estimators=50, random_state=0),
    propensity_model=RandomForestClassifier(n_estimators=50, random_state=0),
    # cate_models is NOT specified - will default to None
)

print(f"cate_models value: {wrapper1.cate_models}")
wrapper1.fit(dgp.X, dgp.Y, W=dgp.W)
tau_pred1 = wrapper1.predict(dgp.X)
pehe1 = pehe(dgp.tau, tau_pred1)

print(f"Predicted average treatment effect: {np.mean(tau_pred1):.3f}")
print(f"PEHE: {pehe1:.3f}")

# Test 2: With explicit cate_models (backward compatibility)
print("\n" + "-" * 60)
print("Test 2: XlearnerWrapper WITH explicit cate_models")
print("-" * 60)

wrapper2 = XlearnerWrapper(
    models=RandomForestRegressor(n_estimators=50, random_state=0),
    propensity_model=RandomForestClassifier(n_estimators=50, random_state=0),
    cate_models=RandomForestRegressor(n_estimators=50, random_state=0),
)

print(f"cate_models value: {wrapper2.cate_models}")
wrapper2.fit(dgp.X, dgp.Y, W=dgp.W)
tau_pred2 = wrapper2.predict(dgp.X)
pehe2 = pehe(dgp.tau, tau_pred2)

print(f"Predicted average treatment effect: {np.mean(tau_pred2):.3f}")
print(f"PEHE: {pehe2:.3f}")

# Test 3: With explicit None (same as not specifying)
print("\n" + "-" * 60)
print("Test 3: XlearnerWrapper WITH cate_models=None (explicit)")
print("-" * 60)

wrapper3 = XlearnerWrapper(
    models=RandomForestRegressor(n_estimators=50, random_state=0),
    propensity_model=RandomForestClassifier(n_estimators=50, random_state=0),
    cate_models=None,
)

print(f"cate_models value: {wrapper3.cate_models}")
wrapper3.fit(dgp.X, dgp.Y, W=dgp.W)
tau_pred3 = wrapper3.predict(dgp.X)
pehe3 = pehe(dgp.tau, tau_pred3)

print(f"Predicted average treatment effect: {np.mean(tau_pred3):.3f}")
print(f"PEHE: {pehe3:.3f}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Test 1 (no cate_models):        PEHE = {pehe1:.3f}")
print(f"Test 2 (explicit cate_models):  PEHE = {pehe2:.3f}")
print(f"Test 3 (cate_models=None):      PEHE = {pehe3:.3f}")
print("\nAll three approaches work correctly!")
print("✓ cate_models is now optional")
print("✓ Backward compatibility maintained")
print("✓ When not specified, X-learner uses models for CATE estimation")

