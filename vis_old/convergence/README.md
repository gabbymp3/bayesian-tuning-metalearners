# Convergence Analysis

This directory contains a Jupyter notebook for analyzing how PEHE and PEHE Plug-in metrics converge as the number of repetitions (R) increases.

## Purpose

The convergence analysis helps determine:
- Whether the number of repetitions (R) is sufficient for stable results
- How quickly different tuners converge to stable performance
- If there are differences in convergence behavior across dimensional settings
- Which combinations of model/tuner/setting require more repetitions

## Files

- `convergence_analysis.ipynb` - Main notebook for convergence visualization
- `convergence_1d.png` - Convergence plots for 1-dimensional setting
- `convergence_2d.png` - Convergence plots for 2-dimensional setting
- `convergence_4d.png` - Convergence plots for 4-dimensional setting
- `convergence_6d.png` - Convergence plots for 6-dimensional setting

## Requirements

The notebook expects data from multiple results folders with different R values:
- `../results/R_5/` - Results with R=5 repetitions
- `../results/R_10/` - Results with R=10 repetitions
- `../results/R_20/` - Results with R=20 repetitions

Each folder should contain CSV files in the format: `{setting}/{model}/raw_results.csv`

Example:
```
results/R_5/
  ├── 1d/x_cb/raw_results.csv
  ├── 1d/x_rf/raw_results.csv
  ├── 2d/x_cb/raw_results.csv
  └── ...
results/R_10/
  ├── 1d/x_cb/raw_results.csv
  └── ...
```

## Usage

1. Ensure you have results folders for multiple R values
2. Open `convergence_analysis.ipynb` in Jupyter
3. Adjust the `r_values` list if needed (default: `[5, 10, 20]`)
4. Run all cells
5. Review the generated plots

## Visualization Details

### Plot Structure
- **4 separate figures**: One for each dimensional setting (1d, 2d, 4d, 6d)
- **2 subplots per figure**: Side-by-side comparison of X_CB and X_RF models
- **X-axis**: Number of repetitions (R)
- **Y-axis**: Error values (PEHE and PEHE Plug-in)

### Visual Encoding
- **Colors** (by tuner):
  - 🔵 Blue (#2E86AB) - Bayesian optimization
  - 🟣 Purple (#A23B72) - Grid search
  - 🟠 Orange (#F18F01) - Random search

- **Line styles** (by metric):
  - **Solid line (—)** with circle markers (○) - PEHE
  - **Dashed line (- -)** with square markers (□) - PEHE Plug-in

### Interpreting the Plots

**Good convergence** is indicated by:
- Lines that flatten out (plateau) as R increases
- Small changes in error values between consecutive R values
- Consistent ordering of tuners across R values

**Poor convergence** suggests:
- Lines that continue to change significantly
- Erratic behavior or crossing lines
- Need for additional repetitions

## Example Insights

From the convergence plots, you can answer questions like:
- "Is R=20 sufficient, or do we need R=30?"
- "Does Bayesian optimization converge faster than random search?"
- "Are higher-dimensional settings more stable or less stable?"
- "Which metric (PEHE vs PEHE Plug-in) shows better convergence?"

## Customization

To modify the analysis:

1. **Change R values**: Edit the `r_values` list in the second code cell
2. **Add more settings**: Modify the `settings` list
3. **Adjust colors**: Update the `tuner_colors` dictionary
4. **Change plot size**: Modify `figsize=(16, 6)` parameter
5. **Adjust line styles**: Change `linestyle`, `linewidth`, or `markersize` parameters

## Notes

- The notebook automatically skips missing data folders/files with warnings
- All plots are saved at 300 DPI for publication quality
- Grid lines are included for easier reading of values
- Legends are positioned automatically to avoid overlapping with data

