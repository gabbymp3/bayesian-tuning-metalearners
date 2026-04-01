from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FOLDER = ROOT / "results_R_30"
OUTPUT_DIR = ROOT / "plots_30" / "pehe_alignment"
TARGET_TUNERS = ["bayes", "random"]
LEARNERS = ["x_cb", "x_rf"]
DIMENSIONS = ["1d", "2d", "4d", "6d"]
TUNER_COLORS = {
    "bayes": "#2A6F97",
    "random": "#C8553D",
}


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan, np.nan
    if method == "pearson":
        result = pearsonr(x, y)
    else:
        result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def load_pehe_results(results_folder: Path) -> pd.DataFrame:
    frames = []

    for raw_path in sorted(results_folder.glob("*/*/raw_results.csv")):
        learner = raw_path.parent.parent.name
        dimension = raw_path.parent.name
        df = pd.read_csv(raw_path)
        df = df[df["tuner"].isin(TARGET_TUNERS)].copy()
        df["learner"] = learner
        df["dimension"] = dimension
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No raw_results.csv files found under {results_folder}")

    return pd.concat(frames, ignore_index=True)


def load_final_tuning_mse(results_folder: Path) -> pd.DataFrame:
    records = []

    for path in sorted(results_folder.glob("*/*/convergence/*/convergence_R*.csv")):
        learner = path.parts[-5]
        dimension = path.parts[-4]
        tuner = path.parts[-2]
        if tuner not in TARGET_TUNERS:
            continue

        rep = int(path.stem.split("R")[-1])
        df = pd.read_csv(path).sort_values("iteration")
        records.append(
            {
                "learner": learner,
                "dimension": dimension,
                "tuner": tuner,
                "rep": rep,
                "final_tuning_mse": float(df["best_so_far"].iloc[-1]),
            }
        )

    if not records:
        raise FileNotFoundError(f"No convergence files found under {results_folder}")

    return pd.DataFrame(records)


def build_rep_level_dataset(results_folder: Path) -> pd.DataFrame:
    pehe_df = load_pehe_results(results_folder)
    mse_df = load_final_tuning_mse(results_folder)
    merged = pehe_df.merge(
        mse_df,
        on=["learner", "dimension", "tuner", "rep"],
        how="inner",
        validate="one_to_one",
    )
    return merged.sort_values(["learner", "dimension", "tuner", "rep"]).reset_index(drop=True)


def build_alignment_stats(rep_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (learner, dimension), subset in rep_df.groupby(["learner", "dimension"]):
        pearson_stat, pearson_p = safe_corr(subset["final_tuning_mse"], subset["pehe"], "pearson")
        spearman_stat, spearman_p = safe_corr(subset["final_tuning_mse"], subset["pehe"], "spearman")

        rows.append(
            {
                "learner": learner,
                "dimension": dimension,
                "scope": "pooled",
                "n_obs": len(subset),
                "pearson_r": pearson_stat,
                "pearson_pvalue": pearson_p,
                "spearman_rho": spearman_stat,
                "spearman_pvalue": spearman_p,
            }
        )

        for tuner, tuner_subset in subset.groupby("tuner"):
            pearson_stat, pearson_p = safe_corr(
                tuner_subset["final_tuning_mse"], tuner_subset["pehe"], "pearson"
            )
            spearman_stat, spearman_p = safe_corr(
                tuner_subset["final_tuning_mse"], tuner_subset["pehe"], "spearman"
            )
            rows.append(
                {
                    "learner": learner,
                    "dimension": dimension,
                    "scope": tuner,
                    "n_obs": len(tuner_subset),
                    "pearson_r": pearson_stat,
                    "pearson_pvalue": pearson_p,
                    "spearman_rho": spearman_stat,
                    "spearman_pvalue": spearman_p,
                }
            )

    return pd.DataFrame(rows).sort_values(["learner", "dimension", "scope"]).reset_index(drop=True)


def build_delta_dataset(rep_df: pd.DataFrame) -> pd.DataFrame:
    bayes = rep_df[rep_df["tuner"] == "bayes"][
        ["learner", "dimension", "rep", "final_tuning_mse", "pehe"]
    ].rename(columns={"final_tuning_mse": "bayes_final_tuning_mse", "pehe": "bayes_pehe"})
    random = rep_df[rep_df["tuner"] == "random"][
        ["learner", "dimension", "rep", "final_tuning_mse", "pehe"]
    ].rename(columns={"final_tuning_mse": "random_final_tuning_mse", "pehe": "random_pehe"})

    paired = bayes.merge(random, on=["learner", "dimension", "rep"], how="inner", validate="one_to_one")
    paired["delta_mse_bayes_minus_random"] = (
        paired["bayes_final_tuning_mse"] - paired["random_final_tuning_mse"]
    )
    paired["delta_pehe_bayes_minus_random"] = paired["bayes_pehe"] - paired["random_pehe"]
    return paired.sort_values(["learner", "dimension", "rep"]).reset_index(drop=True)


def build_delta_stats(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (learner, dimension), subset in delta_df.groupby(["learner", "dimension"]):
        pearson_stat, pearson_p = safe_corr(
            subset["delta_mse_bayes_minus_random"],
            subset["delta_pehe_bayes_minus_random"],
            "pearson",
        )
        spearman_stat, spearman_p = safe_corr(
            subset["delta_mse_bayes_minus_random"],
            subset["delta_pehe_bayes_minus_random"],
            "spearman",
        )

        rows.append(
            {
                "learner": learner,
                "dimension": dimension,
                "n_pairs": len(subset),
                "mean_delta_mse_bayes_minus_random": subset["delta_mse_bayes_minus_random"].mean(),
                "mean_delta_pehe_bayes_minus_random": subset["delta_pehe_bayes_minus_random"].mean(),
                "pearson_r": pearson_stat,
                "pearson_pvalue": pearson_p,
                "spearman_rho": spearman_stat,
                "spearman_pvalue": spearman_p,
                "bayes_better_both": int(
                    ((subset["delta_mse_bayes_minus_random"] < 0) & (subset["delta_pehe_bayes_minus_random"] < 0)).sum()
                ),
                "bayes_better_mse_worse_pehe": int(
                    ((subset["delta_mse_bayes_minus_random"] < 0) & (subset["delta_pehe_bayes_minus_random"] > 0)).sum()
                ),
                "bayes_worse_mse_better_pehe": int(
                    ((subset["delta_mse_bayes_minus_random"] > 0) & (subset["delta_pehe_bayes_minus_random"] < 0)).sum()
                ),
                "random_better_both": int(
                    ((subset["delta_mse_bayes_minus_random"] > 0) & (subset["delta_pehe_bayes_minus_random"] > 0)).sum()
                ),
                "ties_on_any_axis": int(
                    ((subset["delta_mse_bayes_minus_random"] == 0) | (subset["delta_pehe_bayes_minus_random"] == 0)).sum()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["learner", "dimension"]).reset_index(drop=True)


def build_win_rates(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (learner, dimension), subset in delta_df.groupby(["learner", "dimension"]):
        bayes_wins = (subset["delta_pehe_bayes_minus_random"] < 0).sum()
        random_wins = (subset["delta_pehe_bayes_minus_random"] > 0).sum()
        ties = (subset["delta_pehe_bayes_minus_random"] == 0).sum()
        n = len(subset)
        rows.append(
            {
                "learner": learner,
                "dimension": dimension,
                "n_pairs": n,
                "bayes_mean_pehe": subset["bayes_pehe"].mean(),
                "random_mean_pehe": subset["random_pehe"].mean(),
                "bayes_win_count": int(bayes_wins),
                "random_win_count": int(random_wins),
                "tie_count": int(ties),
                "bayes_win_rate": bayes_wins / n,
                "random_win_rate": random_wins / n,
                "tie_rate": ties / n,
            }
        )

    return pd.DataFrame(rows).sort_values(["learner", "dimension"]).reset_index(drop=True)


def plot_mse_vs_pehe(rep_df: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(LEARNERS), len(DIMENSIONS), figsize=(18, 9), sharey=False)

    for row, learner in enumerate(LEARNERS):
        for col, dimension in enumerate(DIMENSIONS):
            ax = axes[row, col]
            subset = rep_df[(rep_df["learner"] == learner) & (rep_df["dimension"] == dimension)]

            for tuner in TARGET_TUNERS:
                tuner_subset = subset[subset["tuner"] == tuner]
                ax.scatter(
                    tuner_subset["final_tuning_mse"],
                    tuner_subset["pehe"],
                    label=tuner.capitalize(),
                    s=40,
                    alpha=0.8,
                    color=TUNER_COLORS[tuner],
                )

            pearson_stat, _ = safe_corr(subset["final_tuning_mse"], subset["pehe"], "pearson")
            spearman_stat, _ = safe_corr(subset["final_tuning_mse"], subset["pehe"], "spearman")

            ax.set_title(f"{learner.upper()} {dimension.upper()}")
            ax.set_xlabel("Final tuning MSE")
            if col == 0:
                ax.set_ylabel("PEHE")
            ax.text(
                0.03,
                0.97,
                f"r = {pearson_stat:.2f}\nρ = {spearman_stat:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Final tuning MSE versus PEHE by setting", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def plot_delta_scatter(delta_df: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(LEARNERS), len(DIMENSIONS), figsize=(18, 9), sharex=False, sharey=False)

    for row, learner in enumerate(LEARNERS):
        for col, dimension in enumerate(DIMENSIONS):
            ax = axes[row, col]
            subset = delta_df[(delta_df["learner"] == learner) & (delta_df["dimension"] == dimension)]

            ax.scatter(
                subset["delta_mse_bayes_minus_random"],
                subset["delta_pehe_bayes_minus_random"],
                s=40,
                alpha=0.85,
                color="#5B8E7D",
            )
            ax.axhline(0, color="#999999", linewidth=1, linestyle="--")
            ax.axvline(0, color="#999999", linewidth=1, linestyle="--")

            same_direction = (
                np.sign(subset["delta_mse_bayes_minus_random"]) == np.sign(subset["delta_pehe_bayes_minus_random"])
            ).mean()
            pearson_stat, _ = safe_corr(
                subset["delta_mse_bayes_minus_random"],
                subset["delta_pehe_bayes_minus_random"],
                "pearson",
            )

            ax.set_title(f"{learner.upper()} {dimension.upper()}")
            ax.set_xlabel("ΔMSE (Bayes - Random)")
            if col == 0:
                ax.set_ylabel("ΔPEHE (Bayes - Random)")
            ax.text(
                0.03,
                0.97,
                f"r = {pearson_stat:.2f}\nagreement = {same_direction:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

    fig.suptitle("Rep-level ΔMSE versus ΔPEHE", fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def plot_win_rates(win_df: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(LEARNERS), figsize=(12, 4.8), sharey=True)

    if len(LEARNERS) == 1:
        axes = [axes]

    for ax, learner in zip(axes, LEARNERS):
        subset = win_df[win_df["learner"] == learner].copy()
        subset["setting"] = subset["dimension"].str.upper()
        ax.bar(subset["setting"], subset["bayes_win_rate"], color="#2A6F97", alpha=0.9)
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{learner.upper()}")
        ax.set_xlabel("Setting")
        ax.set_ylabel("Bayes PEHE win rate")

        for _, row in subset.iterrows():
            ax.text(
                row["setting"],
                row["bayes_win_rate"] + 0.03,
                f"{row['bayes_win_count']}/{row['n_pairs']}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Bayes versus Random PEHE win rates", fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rep_df = build_rep_level_dataset(RESULTS_FOLDER)
    rep_df.to_csv(OUTPUT_DIR / "rep_level_mse_pehe.csv", index=False)

    alignment_stats = build_alignment_stats(rep_df)
    alignment_stats.to_csv(OUTPUT_DIR / "mse_vs_pehe_alignment_stats.csv", index=False)

    delta_df = build_delta_dataset(rep_df)
    delta_df.to_csv(OUTPUT_DIR / "delta_mse_delta_pehe.csv", index=False)

    delta_stats = build_delta_stats(delta_df)
    delta_stats.to_csv(OUTPUT_DIR / "delta_mse_vs_delta_pehe_stats.csv", index=False)

    win_df = build_win_rates(delta_df)
    win_df.to_csv(OUTPUT_DIR / "pehe_win_rates.csv", index=False)

    plot_mse_vs_pehe(rep_df, OUTPUT_DIR / "mse_vs_pehe_by_setting.png")
    plot_delta_scatter(delta_df, OUTPUT_DIR / "delta_mse_vs_delta_pehe.png")
    plot_win_rates(win_df, OUTPUT_DIR / "pehe_win_rates.png")

    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("\nWin rates:")
    print(win_df.to_string(index=False))


if __name__ == "__main__":
    main()
