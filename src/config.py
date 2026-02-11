from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from skopt.space import Integer
from src.tuning import grid_search, random_search, bayesian_search


R = 5
dgp_params = {"N": 1000, "d": 15, "alpha": 0.5}


# -------------------------
# RANDOM FOREST CONFIG
# -------------------------
rf_config = {
    "name": "x_rf",
    "models": RandomForestRegressor(random_state=0),
    "propensity_model": RandomForestClassifier(random_state=0),
}

rf_tuners = [
    {
        "name": "grid",
        "fn": grid_search,
        "param_grid": {
            "models__n_estimators": [50, 100],
            "models__max_depth": [3, 5, 7]
        },
        "kwargs": {"cv": 3}
    },
    {
        "name": "random",
        "fn": random_search,
        "param_dist": {
            "models__n_estimators": Integer(20, 150),
            "models__max_depth": Integer(3, 10)
        },
        "kwargs": {"cv": 3, "n_iter": 10}
    },
    {
        "name": "bayes",
        "fn": bayesian_search,
        "param_dist": {
            "models__n_estimators": Integer(20, 150),
            "models__max_depth": Integer(3, 10)
        },
        "kwargs": {"cv": 3, "n_iter": 10}
    },
]


# -------------------------
# CATBOOST CONFIG
# -------------------------
cb_config = {
    "name": "x_cb",
    "models": CatBoostRegressor(verbose=0, random_state=0),
    "propensity_model": CatBoostClassifier(verbose=0, random_state=0),
}

cb_tuners = [
    {
        "name": "grid",
        "fn": grid_search,
        "param_grid": {
            "models__depth": [4, 6, 8],
            "models__iterations": [100, 200]
        },
        "kwargs": {"cv": 3}
    },
    {
        "name": "random",
        "fn": random_search,
        "param_dist": {
            "models__depth": Integer(3, 10),
            "models__iterations": Integer(50, 300)
        },
        "kwargs": {"cv": 3, "n_iter": 10}
    },
    {
        "name": "bayes",
        "fn": bayesian_search,
        "param_dist": {
            "models__depth": Integer(3, 10),
            "models__iterations": Integer(50, 300)
        },
        "kwargs": {"cv": 3, "n_iter": 10}
    },
]
