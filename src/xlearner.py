from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
import numpy as np


class XlearnerWrapper(BaseEstimator):
    '''
    Wrapper for EconML Xlearner to make it compatible with sklearn API for HP tuning.
    '''
    def __init__(self, models, propensity_model, cate_models):
        self.models = models
        self.propensity_model = propensity_model
        self.cate_models = cate_models
        self._est = None
        #self.true_tau = None
    
    '''
    def get_params(self, deep=True):
        return {"est": self._est}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
    '''

    
    
    def fit(self, Y, W, X=None):
        self._est = XLearner(
            models=self.models,
            propensity_model=self.propensity_model,
            cate_models=self.cate_models,
        )
        self._est.fit(Y, W, X=X)
        return self
    
    def predict(self, X):
        # CATE prediction, tau_hat
        return self._est.effect(X)
    
    def predict_outcome(self, X, W):
        # factual outcome prediction y_hat 
        X = np.asarray(X)
        W = np.asarray(W).reshape(-1)
        assert X.shape[0] == W.shape[0], "X and W must have same number of rows"

        y_hat = np.empty(W.shape[0], dtype=float)
        y_hat[W == 0] = self._est.models[0].predict(X[W == 0])
        y_hat[W == 1] = self._est.models[1].predict(X[W == 1])

        return y_hat

