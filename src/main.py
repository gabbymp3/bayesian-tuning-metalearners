import os
import importlib
import pandas as pd
import warnings

from src.experiment import run_experiment
from src.dgp import simulate_dataset

warnings.filterwarnings("ignore")


# -------------------------------------------------
# Select which experiments to run
# -------------------------------------------------

CONFIGS_TO_RUN = [
    "config_1d",
    "config_2d",
    "config_4d",
    "config_6d",
]


# -------------------------------------------------
# Run experiments
# -------------------------------------------------

for config_name in CONFIGS_TO_RUN:
    print("\n" + "="*80)
    print(f"\nRunning experiment: {config_name}")
    print("="*80)

    module = importlib.import_module(
        f"src.experiment_configs.{config_name}"
    )

    config = module.get_config()

    for learner in config["learners"]:

        print(f"  → Learner: {learner['name']}")
        print("-"*80)

        summary, raw_results = run_experiment(
            learner_config={
                "name": learner["name"],
                "models": learner["models"],
                "propensity_model": learner["propensity_model"],
            },
            tuners=learner["tuners"],
            R=config["R"],
            simulate_dataset_fn=simulate_dataset,
            dgp_params=config["dgp_params"],
            base_seed=config["base_seed"],
        )

        # -------------------------------------------------
        # Create output directory
        # -------------------------------------------------

        output_dir = os.path.join(
            "results",
            f"R_{config['R']}",
            learner["name"],
            config["name"],
        )
        os.makedirs(output_dir, exist_ok=True)

        # -------------------------------------------------
        # Save summary
        # -------------------------------------------------

        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

        # -------------------------------------------------
        # Save raw results
        # -------------------------------------------------

        raw_rows = []
        for tuner_name, metrics in raw_results.items():
            for r, (pehe, pehe_plug) in enumerate(
                zip(metrics["pehe"], metrics["pehe_plug"])
            ):
                raw_rows.append({
                    "learner": learner["name"],
                    "tuner": tuner_name,
                    "rep": r,
                    "pehe": pehe,
                    "pehe_plug": pehe_plug,
                })

        raw_df = pd.DataFrame(raw_rows)
        raw_path = os.path.join(output_dir, "raw_results.csv")
        raw_df.to_csv(raw_path, index=False)

        print(f"  Saved results of {learner['name']} to {output_dir}")
    print(f"Experiment {config_name} completed.")