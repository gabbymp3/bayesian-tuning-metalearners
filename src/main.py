import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from experiment import run_experiment
from dgp import simulate_dataset
from config import (
    R,
    dgp_params,
    rf_config,
    rf_tuners,
    cb_config,
    cb_tuners
)

warnings.filterwarnings("ignore")

output_dir = "experiment_results"
os.makedirs(output_dir, exist_ok=True)

all_summaries = []
all_raw_rows = []


# ----------------------------------
# RUN RF EXPERIMENT
# ----------------------------------
rf_summary, rf_raw = run_experiment(
    learner_config=rf_config,
    tuners=rf_tuners,
    R=R,
    simulate_dataset_fn=simulate_dataset,
    dgp_params=dgp_params,
    base_seed=42
)

# ----------------------------------
# RUN CATBOOST EXPERIMENT
# ----------------------------------
cb_summary, cb_raw = run_experiment(
    learner_config=cb_config,
    tuners=cb_tuners,
    R=R,
    simulate_dataset_fn=simulate_dataset,
    dgp_params=dgp_params,
    base_seed=42
)


# ----------------------------------
# Combine results
# ----------------------------------
all_summaries.extend(rf_summary)
all_summaries.extend(cb_summary)

for learner_raw, learner_name in [
    (rf_raw, "x_rf"),
    (cb_raw, "x_cb"),
]:
    for tuner_name, metrics in learner_raw.items():
        for r, (pehe, pehe_plug) in enumerate(
            zip(metrics["pehe"], metrics["pehe_plug"])
        ):
            all_raw_rows.append({
                "learner": learner_name,
                "tuner": tuner_name,
                "rep": r,
                "pehe": pehe,
                "pehe_plug": pehe_plug
            })


summary_df = pd.DataFrame(all_summaries)
raw_df = pd.DataFrame(all_raw_rows)

summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
raw_df.to_csv(os.path.join(output_dir, "raw_results.csv"), index=False)


# ----------------------------------
# Plot
# ----------------------------------
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.boxplot(data=raw_df, x="tuner", y="pehe", hue="learner")
plt.title("PEHE Distribution")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pehe_boxplot.png"))
plt.close()

print("Experiments completed successfully.")
