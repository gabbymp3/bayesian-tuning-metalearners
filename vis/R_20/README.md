# Boxplots for R=20 Results

This folder contains a Jupyter notebook for creating boxplots from the experimental results in the `results/R_20` folder.

## Contents

- `create_boxplots.ipynb` - Main notebook for generating all boxplots and statistical summaries

## What the Notebook Does

The notebook analyzes results from experiments with **R=20 repetitions** across different dimensional settings:
- **1d** - 1-dimensional setting
- **2d** - 2-dimensional setting  
- **4d** - 4-dimensional setting
- **6d** - 6-dimensional setting

For each setting, it compares:
- **Models**: `x_cb` (CatBoost) and `x_rf` (Random Forest)
- **Tuners**: `grid`, `random`, and `bayes`

## Generated Outputs

The notebook generates the following visualizations:

1. **pehe_boxplot_r20.png** - PEHE boxplots for all 4 dimensional settings (2x2 grid)
2. **pehe_plug_boxplot_r20.png** - PEHE plug-in boxplots for all 4 dimensional settings (2x2 grid)
3. **1d_boxplot_r20.png** - Detailed boxplots for 1d setting (PEHE and PEHE plug-in side by side)
4. **2d_boxplot_r20.png** - Detailed boxplots for 2d setting
5. **4d_boxplot_r20.png** - Detailed boxplots for 4d setting
6. **6d_boxplot_r20.png** - Detailed boxplots for 6d setting
7. **dimension_comparison_r20.png** - Comparison across all dimensions
8. **summary_statistics_r20.csv** - Comprehensive statistical summary table

## How to Run

1. Make sure you have the required packages installed:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

2. Open the notebook:
   ```bash
   jupyter notebook create_boxplots.ipynb
   ```

3. Run all cells to generate the visualizations

## Color Scheme

The boxplots use the following color scheme for tuners:
- **Grid Search**: Blue (#2E86AB)
- **Random Search**: Purple (#A23B72)
- **Bayesian Optimization**: Orange (#F18F01)

## Data Source

All data is loaded from the `../results/R_20/` directory, which contains raw results from experiments with 20 repetitions for each configuration.

