
from sklearn.model_selection import KFold
import numpy as np


def outcome_mse(estimator, X, Y, W):
    """
    Factual outcome MSE
    """
    y_hat = estimator.predict_outcome(X, W)
    return np.mean((Y - y_hat) ** 2)


'''
def make_outcome_mse_scorer(W):
    def scorer(estimator, X, y_true):
        y_pred = estimator.predict_outcome(X, W)
        return -np.mean((y_true - y_pred) ** 2)
    return scorer

def outcome_mse_scorer(estimator, X, y_true, *, W):
    assert hasattr(estimator, "predict_outcome")
    y_pred = estimator.predict_outcome(X, W)
    return -np.mean((y_true - y_pred) ** 2)


mse_scorer = make_scorer(
    outcome_mse_scorer,
    greater_is_better=False,
    needs_estimator=True
)

# custom MSE function to be used during tuning
def outcome_mse_scorer(estimator, X, Y, W=None):
    assert hasattr(estimator, "predict_outcome")
    y_pred = estimator.predict_outcome(X, W)
    return np.mean((Y - y_pred) ** 2)

mse_scorer = make_scorer(outcome_mse_scorer, greater_is_better=False)
'''



# PEHE and PEHE plugin to be used after tuning

def pehe(tau_true_or_plug, tau_hat):
    return np.mean((tau_true_or_plug - tau_hat) ** 2)



# custom cross_predict function to be compatible with XlearnerWrapper

def cross_predict_tau(estimator_type, estimator_params, X, Y, W=None, cv=5):
    '''
    Custom cross-prediction function for XlearnerWrapper.

    1. Splits data into k folds.
    2. For each fold, fits the estimator on the k-1 folds and computes out-of-fold predictions.
    3. Averages the out-of-fold predictions for each observation.
    '''

    N = len(Y)
    kfold = KFold(n_splits=cv, shuffle=True, random_state=123)
    tau_oof = np.zeros(N)

    for fold, (train_idx, _) in enumerate(kfold.split(X)):
        est = estimator_type(**estimator_params)
        #est.fit(Y[train_idx], W[train_idx], X[train_idx])
        est.fit(X[train_idx], Y[train_idx], **{"W": W[train_idx]})
        tau_fold = est.predict(X)

        tau_oof += tau_fold / cv
    
    return tau_oof
    
