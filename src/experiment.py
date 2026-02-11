import numpy as np
from sklearn.model_selection import KFold
from src.metrics_helpers import pehe, cross_predict_tau
from src.xlearner import XlearnerWrapper

def run_experiment(
    learners,
    tuners,
    R,
    simulate_dataset_fn,
    dgp_params,
    base_seed=0,
    cv_plug=5  # folds for cross-predict
):
    """
    Run repeated experiments for tuned X-learners and evaluate PEHE metrics.

    Returns
    -------
    summary : list of dict
        Aggregated PEHE results.
    raw_results : dict
        Per-repetition PEHE values.
    """

    # ---------------------------
    # Storage
    # ---------------------------
    raw_results = {
        (L["name"], T["name"]): {
            "pehe": np.zeros(R),
            "pehe_plug": np.zeros(R),
        }
        for L in learners
        for T in tuners
    }

    # ---------------------------
    # Monte Carlo loop
    # ---------------------------
    for r in range(R):
        # Unpack everything from new simulate_dataset
        X, W, Y, mu0, mu1, Y0, Y1, tau, e = simulate_dataset_fn(
            dgp_params, seed=base_seed + r
        )

        tau_true = tau

        for L in learners:
            base_estimator = XlearnerWrapper(
                models=L["models"],
                propensity_model=L["propensity_model"],
                cate_models=L.get("cate_models", None),  # Optional, defaults to None (uses models)
            )

            for Tu in tuners:
                # -------------------------------
                # Tune using correct argument
                # -------------------------------
                if Tu["fn"].__name__ == "grid_search":
                    best_estimator, best_params, best_score = Tu["fn"](
                        estimator=base_estimator,
                        param_grid=Tu["param_grid"],
                        X=X,
                        Y=Y,
                        W=W,
                        **Tu.get("kwargs", {})
                    )
                elif Tu["fn"].__name__ in ["random_search", "bayesian_search"]:
                    best_estimator, best_params, best_score = Tu["fn"](
                        estimator=base_estimator,
                        param_dist=Tu["param_dist"],
                        X=X,
                        Y=Y,
                        W=W,
                        **Tu.get("kwargs", {})
                    )
                
                else:
                    raise ValueError(f"Unknown tuning function {Tu['fn']}")

                # -------------------------------
                # Estimate CATE
                # -------------------------------
                tau_hat = best_estimator.predict(X)

                # -------------------------------
                # Compute plug-in tau using cross-prediction
                # -------------------------------
                estimator_type = type(best_estimator)
                estimator_params = best_estimator.get_params()
                tau_plug = cross_predict_tau(
                    estimator_type,
                    estimator_params,
                    X, Y, W=W,
                    cv=cv_plug
                )

                key = (L["name"], Tu["name"])
                raw_results[key]["pehe"][r] = pehe(tau_true, tau_hat)
                raw_results[key]["pehe_plug"][r] = pehe(tau_plug, tau_hat)

    # ---------------------------
    # Aggregate results
    # ---------------------------
    summary = []
    for (learner_name, tuner_name), d in raw_results.items():
        pe = d["pehe"]
        pp = d["pehe_plug"]

        summary.append({
            "learner": learner_name,
            "tuner": tuner_name,
            "pehe_mean": float(pe.mean()),
            "pehe_var": float(pe.var(ddof=1)) if len(pe) > 1 else 0.0,
            "pehe_plug_mean": float(pp.mean()),
            "pehe_plug_var": float(pp.var(ddof=1)) if len(pp) > 1 else 0.0,
        })

    return summary, raw_results
