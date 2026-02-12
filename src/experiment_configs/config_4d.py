from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from skopt.space import Integer, Real
from src.tuning import random_search, bayesian_search


R = 5
dgp_params = {"N": 1000, "d": 15, "alpha": 0.5}


# -------------------------
# RANDOM FOREST CONFIG
# -------------------------
rf_config = {
    "name": "x_rf",
    "models": RandomForestRegressor(random_state=0),
    "propensity_model": LogisticRegression(),
}

rf_tuners = [
    {
        "name": "random",
        "fn": random_search,
        "param_dist": {
            "models__n_estimators": Integer(20, 150),
            "models__max_depth": Integer(3, 10),
            "models__min_samples_leaf": Integer(1, 20),
            "models__min_samples_split": Integer(2, 20)
        },
        "kwargs": {"cv": 3, "n_iter": 40}
    },
    {
        "name": "bayes",
        "fn": bayesian_search,
        "param_dist": {
            "models__n_estimators": Integer(20, 150),
            "models__max_depth": Integer(3, 10),
            "models__min_samples_leaf": Integer(1, 20),
            "models__min_samples_split": Integer(2, 20)
        },
        "kwargs": {"cv": 3, "n_iter": 40}
    },
]


# -------------------------
# CATBOOST CONFIG
# -------------------------
cb_config = {
    "name": "x_cb",
    "models": CatBoostRegressor(verbose=0, random_state=0),
    "propensity_model": LogisticRegression(),
}

cb_tuners = [
    {
        "name": "random",
        "fn": random_search,
        "param_dist": {
            "models__learning_rate": Real(0.01, 0.15),
            "models__depth": Integer(3, 10),
            "models__iterations": Integer(50, 300),
            "models__l2_leaf_reg": Integer(1, 10)
        },
        "kwargs": {"cv": 3, "n_iter": 40}
    },
    {
        "name": "bayes",
        "fn": bayesian_search,
        "param_dist": {
            "models__learning_rate": Real(0.01, 0.15),
            "models__depth": Integer(3, 10),
            "models__iterations": Integer(50, 300),
            "models__l2_leaf_reg": Integer(1, 10)
        },
        "kwargs": {"cv": 3, "n_iter": 40}
    },
]
