from sklearn.model_selection import KFold
import numpy as np


def outcome_mse(estimator, X, Y, W):
    """
    Factual outcome MSE
    """
    y_hat = estimator.predict_outcome(X, W)
    mask = np.zeros_like(Y, dtype=bool)

    if np.any(W == 0):
        y_hat[W == 0] = estimator._est.models[0].predict(X[W == 0])
        mask[W == 0] = True

    if np.any(W == 1):
        y_hat[W == 1] = estimator._est.models[1].predict(X[W == 1])
        mask[W == 1] = True

    if not np.any(mask):
        return np.nan

    return np.mean((Y[mask] - y_hat[mask]) ** 2)



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
        est.fit(X[train_idx], Y[train_idx], **{"W": W[train_idx]})
        tau_fold = est.predict(X)

        tau_oof += tau_fold / cv
    
    return tau_oof