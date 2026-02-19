import numpy as np
from src.metrics_helpers import pehe, cross_predict_tau
from src.xlearner import XlearnerWrapper


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

    print("Monte Carlo simulation starting...")
    for r in range(R):
        print(f"  → Repetition {r}/{R}")
        
        # Training dataset (80%)
        X_train, W_train, Y_train, mu0_train, mu1_train, Y0_train, Y1_train, tau_train, e_train = simulate_dataset_fn(
            dgp_params, seed=base_seed + r
        )

        # Test dataset (20%)
        dgp_params_test = dgp_params.copy()
        dgp_params_test["N"] = dgp_params["N"] // 4
        X_test, W_test, Y_test, mu0_test, mu1_test, Y0_test, Y1_test, tau_test, e_test = simulate_dataset_fn(
            dgp_params_test, seed=base_seed + 1000 + r  # offset seed
        )

        # Tune on train set
        tau_true = tau_train

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
                    X=X_train, Y=Y_train, W=W_train,
                    **tuner.get("kwargs", {})
                )
            else:
                best_estimator, best_params, best_score = tuner["fn"](
                    estimator=base_estimator,
                    param_dist=tuner["param_dist"],
                    X=X_train, Y=Y_train, W=W_train,
                    **tuner.get("kwargs", {})
                )

            # Evaluate on test set
            tau_hat = best_estimator.predict(X_test)

            estimator_type = type(best_estimator)
            estimator_params = best_estimator.get_params()

            tau_plug = cross_predict_tau(
                estimator_type,
                estimator_params,
                X_test, Y_test, W=W_test,
                cv=cv_plug
            )

            raw_results[tuner["name"]]["pehe"][r] = pehe(tau_test, tau_hat)
            raw_results[tuner["name"]]["pehe_plug"][r] = pehe(tau_plug, tau_hat)

        print('*'*20 + "Tuning complete." + '*'*20)

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
    print("Monte Carlo simulation complete.")
    return summary, raw_results