import numpy as np
from metrics_helpers import pehe, cross_predict_tau
from xlearner import XlearnerWrapper


def run_experiment(
    learner_config,
    tuners,
    R,
    simulate_dataset_fn,
    dgp_params,
    base_seed=0,
    cv_plug=5
):

    """
    Run repeated experiments for an X-learner across multiple tuners and evaluate PEHE metrics.

    Returns
    -------
    summary : list of dict
        Aggregated PEHE results.
    raw_results : dict
        Per-repetition PEHE values.
    """

    learner_name = learner_config["name"]

    raw_results = {
        tuner["name"]: {
            "pehe": np.zeros(R),
            "pehe_plug": np.zeros(R),
        }
        for tuner in tuners
    }

    for r in range(R):

        X, W, Y, mu0, mu1, Y0, Y1, tau, e = simulate_dataset_fn(
            dgp_params, seed=base_seed + r
        )

        tau_true = tau

        base_estimator = XlearnerWrapper(
            models=learner_config["models"],
            propensity_model=learner_config["propensity_model"],
        )

        for tuner in tuners:

            # Select correct argument name
            if tuner["fn"].__name__ == "grid_search":
                best_estimator, best_params, best_score = tuner["fn"](
                    estimator=base_estimator,
                    param_grid=tuner["param_grid"],
                    X=X, Y=Y, W=W,
                    **tuner.get("kwargs", {})
                )
            else:
                best_estimator, best_params, best_score = tuner["fn"](
                    estimator=base_estimator,
                    param_dist=tuner["param_dist"],
                    X=X, Y=Y, W=W,
                    **tuner.get("kwargs", {})
                )

            tau_hat = best_estimator.predict(X)

            estimator_type = type(best_estimator)
            estimator_params = best_estimator.get_params()

            tau_plug = cross_predict_tau(
                estimator_type,
                estimator_params,
                X, Y, W=W,
                cv=cv_plug
            )

            raw_results[tuner["name"]]["pehe"][r] = pehe(tau_true, tau_hat)
            raw_results[tuner["name"]]["pehe_plug"][r] = pehe(tau_plug, tau_hat)

    # Aggregate
    summary = []
    for tuner_name, metrics in raw_results.items():
        summary.append({
            "learner": learner_name,
            "tuner": tuner_name,
            "pehe_mean": float(metrics["pehe"].mean()),
            "pehe_var": float(metrics["pehe"].var(ddof=1)),
            "pehe_plug_mean": float(metrics["pehe_plug"].mean()),
            "pehe_plug_var": float(metrics["pehe_plug"].var(ddof=1)),
        })

    return summary, raw_results