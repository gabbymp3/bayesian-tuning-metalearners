# Aggregate Results Notebook - Updated

## Changes Made

The `aggregate_results.ipynb` notebook has been updated to:

### 1. **Read from `raw_results.csv` instead of `summary.csv`**
   - The notebook now loads individual repetition data from `raw_results.csv` files
   - This allows for proper calculation of standard deviations across repetitions

### 2. **Calculate Standard Deviations**
   - **Before**: Used variance from summary files (which was incorrect)
   - **After**: Calculates proper standard deviation from raw repetition data
   - Columns now include:
     - `pehe_mean`: Mean PEHE across repetitions
     - `pehe_std`: Standard deviation of PEHE (calculated from raw data)
     - `pehe_plug_mean`: Mean plug-in PEHE across repetitions
     - `pehe_plug_std`: Standard deviation of plug-in PEHE (calculated from raw data)
     - `n_reps`: Number of repetitions (should be 5 or 10)

### 3. **Generate Comparison Tables**
   - Creates separate comparison tables for each **setting × model × tuner** combination
   - Each table shows R=5 vs R=10 side-by-side for easy comparison
   - Format:
     ```
     Setting: 1D | Model: X_RF
     --- Tuner: RANDOM ---
                pehe_mean  pehe_std  pehe_plug_mean  pehe_plug_std  n_reps
     R                                                                      
     5           0.1234     0.0123    0.2345          0.0234         5
     10          0.1200     0.0110    0.2300          0.0220         10
     ```

### 4. **Additional Features**
   - Export aggregated results to `aggregated_results.csv`
   - Summary of best/worst performing configurations
   - Clear organization by setting (1d, 2d, 4d, 6d) and model (x_rf, x_cb)

## How to Use

1. **Open the notebook** in Jupyter:
   ```bash
   cd draft
   jupyter notebook aggregate_results.ipynb
   ```

2. **Run all cells** to:
   - Load raw results from `results_5/` and `results_10/`
   - Calculate proper statistics with standard deviations
   - Generate comparison tables for all combinations
   - Export aggregated results

3. **Review comparison tables** to see:
   - How performance changes between R=5 and R=10
   - Which tuning method (random vs bayes) performs better
   - How different settings and models compare

## Expected Output

The notebook will generate:
- **Comparison tables**: One for each setting/model/tuner combination (4 settings × 2 models × 2-3 tuners = ~24 tables)
- **Aggregated CSV**: `aggregated_results.csv` with all statistics
- **Best configurations**: Summary of top and bottom performers

## Data Structure

Input files expected:
```
results_5/
  ├── 1d/
  │   ├── x_rf/raw_results.csv
  │   └── x_cb/raw_results.csv
  ├── 2d/
  │   ├── x_rf/raw_results.csv
  │   └── x_cb/raw_results.csv
  ...

results_10/
  ├── 1d/
  │   ├── x_rf/raw_results.csv
  │   └── x_cb/raw_results.csv
  ...
```

Each `raw_results.csv` contains:
- `learner`: Model name (e.g., 'x_rf', 'x_cb')
- `tuner`: Tuning method (e.g., 'random', 'bayes', 'grid')
- `pehe`: PEHE value for each repetition
- `pehe_plug`: Plug-in PEHE value for each repetition
- Additional columns (rep_id, best_params, etc.)

## Key Differences from Previous Version

| Aspect | Before | After |
|--------|--------|-------|
| Data source | `summary.csv` | `raw_results.csv` |
| Statistics | Pre-calculated variance | Calculated std dev from raw data |
| Comparison | Single aggregated table | Separate tables per combination |
| Focus | Overall summary | R=5 vs R=10 comparison |

