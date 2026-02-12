from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from skopt.space import Integer, Real
from src.tuning import grid_search, random_search, bayesian_search

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor
from skopt.space import Integer, Real

from src.tuning import grid_search, random_search, bayesian_search


def get_config():

    R = 5
    dgp_params = {"N": 1000, "d": 15, "alpha": 0.5}

    learners = [
        {
            "name": "x_rf",
            "models": RandomForestRegressor(random_state=0),
            "propensity_model": LogisticRegression(),
            "tuners": [
                {
                    "name": "grid",
                    "fn": grid_search,
                    "param_grid": {
                        "models__n_estimators": [20, 50, 80, 110, 140]
                    },
                    "kwargs": {"cv": 3}
                },
                {
                    "name": "random",
                    "fn": random_search,
                    "param_dist": {
                        "models__n_estimators": Integer(20, 150)
                    },
                    "kwargs": {"cv": 3, "n_iter": 10}
                },
                {
                    "name": "bayes",
                    "fn": bayesian_search,
                    "param_dist": {
                        "models__n_estimators": Integer(20, 150)
                    },
                    "kwargs": {"cv": 3, "n_iter": 10}
                },
            ]
        },
        {
            "name": "x_cb",
            "models": CatBoostRegressor(verbose=0, random_state=0),
            "propensity_model": LogisticRegression(),
            "tuners": [
                {
                    "name": "grid",
                    "fn": grid_search,
                    "param_grid": {
                        "models__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15]
                    },
                    "kwargs": {"cv": 3}
                },
                {
                    "name": "random",
                    "fn": random_search,
                    "param_dist": {
                        "models__learning_rate": Real(0.01, 0.15)
                    },
                    "kwargs": {"cv": 3, "n_iter": 10}
                },
                {
                    "name": "bayes",
                    "fn": bayesian_search,
                    "param_dist": {
                        "models__learning_rate": Real(0.01, 0.15)
                    },
                    "kwargs": {"cv": 3, "n_iter": 10}
                },
            ]
        }
    ]

    return {
        "name": "1d",
        "R": R,
        "dgp_params": dgp_params,
        "learners": learners,
        "base_seed": 42
    }