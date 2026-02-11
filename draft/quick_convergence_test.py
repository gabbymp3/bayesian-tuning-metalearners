"""
Quick test script for tuning convergence visualization.
This is a faster version for testing (fewer iterations).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from skopt import Optimizer
from skopt.space import Integer
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint

from src.dgp import SimulatedDataset
from src.xlearner import XlearnerWrapper
from src.tuning import evaluate_params_cv

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

print("="*60)
print("QUICK CONVERGENCE TEST")
print("="*60)

# 1. Generate Dataset
print("\n1. Generating dataset...")
dgp = SimulatedDataset(N=300, d=10, alpha=0.5, seed=42)
print(f"   Dataset: N={dgp.N}, d={dgp.d}")
print(f"   Treatment: {np.mean(dgp.W):.2%} treated")

# 2. Setup
print("\n2. Setting up search...")
base_estimator = XlearnerWrapper(
    models=RandomForestRegressor(random_state=0),
    propensity_model=RandomForestClassifier(n_estimators=30, random_state=0)
)

param_dist_random = {
    'models__n_estimators': randint(20, 100),
    'models__max_depth': randint(3, 12),
    'models__min_samples_leaf': randint(1, 15)
}

param_dist_bayes = {
    'models__n_estimators': Integer(20, 100),
    'models__max_depth': Integer(3, 12),
    'models__min_samples_leaf': Integer(1, 15)
}

n_iter = 15  # Quick test with fewer iterations
cv = 2       # Fewer CV folds for speed
random_state = 42

# 3. Random Search
print(f"\n3. Running Random Search ({n_iter} iterations)...")
random_scores = []
random_best_scores = []
best_score_random = np.inf

for i, params in enumerate(ParameterSampler(param_dist_random, n_iter=n_iter, random_state=random_state)):
    score = evaluate_params_cv(base_estimator, params, dgp.X, dgp.Y, dgp.W, cv=cv, random_state=random_state)
    random_scores.append(score)
    if score < best_score_random:
        best_score_random = score
    random_best_scores.append(best_score_random)
    print(f"   Iter {i+1}/{n_iter}: MSE={score:.4f}, Best={best_score_random:.4f}")

# 4. Bayesian Optimization
print(f"\n4. Running Bayesian Optimization ({n_iter} iterations)...")
param_names = list(param_dist_bayes.keys())
dimensions = list(param_dist_bayes.values())

optimizer = Optimizer(
    dimensions=dimensions,
    base_estimator="GP",
    acq_func="EI",
    random_state=random_state
)

bayes_scores = []
bayes_best_scores = []
best_score_bayes = np.inf

for i in range(n_iter):
    x = optimizer.ask()
    params = dict(zip(param_names, x))
    score = evaluate_params_cv(base_estimator, params, dgp.X, dgp.Y, dgp.W, cv=cv, random_state=random_state)
    optimizer.tell(x, score)
    bayes_scores.append(score)
    if score < best_score_bayes:
        best_score_bayes = score
    bayes_best_scores.append(best_score_bayes)
    print(f"   Iter {i+1}/{n_iter}: MSE={score:.4f}, Best={best_score_bayes:.4f}")

# 5. Visualize
print("\n5. Creating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Convergence plot
ax1 = axes[0]
iterations = np.arange(1, n_iter + 1)
ax1.plot(iterations, random_best_scores, 'o-', label='Random Search', 
         color='#2E86AB', linewidth=2, markersize=6, alpha=0.7)
ax1.plot(iterations, bayes_best_scores, 's-', label='Bayesian Optimization', 
         color='#A23B72', linewidth=2, markersize=6, alpha=0.7)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Best MSE (lower is better)', fontsize=12)
ax1.set_title('Convergence: Best Score Over Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Exploration plot
ax2 = axes[1]
ax2.scatter(iterations, random_scores, label='Random Search', 
            color='#2E86AB', alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
ax2.scatter(iterations, bayes_scores, label='Bayesian Optimization', 
            color='#A23B72', alpha=0.6, s=80, marker='s', edgecolors='black', linewidth=0.5)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('MSE', fontsize=12)
ax2.set_title('Exploration: All Evaluated Scores', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quick_convergence_test.png', dpi=150, bbox_inches='tight')
print("   Saved as 'quick_convergence_test.png'")

# 6. Summary
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nRandom Search:")
print(f"  Best MSE:  {best_score_random:.4f}")
print(f"  Mean MSE:  {np.mean(random_scores):.4f}")
print(f"  Std MSE:   {np.std(random_scores):.4f}")

print(f"\nBayesian Optimization:")
print(f"  Best MSE:  {best_score_bayes:.4f}")
print(f"  Mean MSE:  {np.mean(bayes_scores):.4f}")
print(f"  Std MSE:   {np.std(bayes_scores):.4f}")

improvement = ((best_score_random - best_score_bayes) / best_score_random) * 100
print(f"\nImprovement: {improvement:.2f}% {'(Bayesian better)' if improvement > 0 else '(Random better)'}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

