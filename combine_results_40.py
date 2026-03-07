#!/usr/bin/env python3
"""
Script to combine results from R_5 and R_35 folders into a new R_40 folder.
This creates 40 Monte Carlo repetitions for each model, tuner, and settings combination.
"""

import pandas as pd
import os
from pathlib import Path
import shutil

def combine_results():
    """Combine results from R_5 and R_35 into R_40."""
    
    # Define source folders
    source_folders = {
        'results_R_5': {
            'x_cb': ['1d', '2d', '4d', '6d'],
            'x_rf': ['1d', '2d', '4d', '6d']
        },
        'results_R_35': {
            'x_cb': ['1d', '2d', '4d', '6d'],
            'x_rf': ['1d', '2d', '4d', '6d']
        }
    }
    
    # Create output directory
    output_base = Path('results_R_40')
    output_base.mkdir(exist_ok=True)
    
    # Process each model type
    for model in ['x_cb', 'x_rf']:
        model_output = output_base / model
        model_output.mkdir(exist_ok=True)
        
        # Process each dimensional setting
        for setting in ['1d', '2d', '4d', '6d']:
            setting_output = model_output / setting
            setting_output.mkdir(exist_ok=True)
            
            print(f"Processing {model}/{setting}...")
            
            # Combine raw_results.csv
            raw_dfs = []
            
            # Load R_5 results (reps 0-4)
            r5_path = Path('results_R_5') / model / setting / 'raw_results.csv'
            if r5_path.exists():
                df_r5 = pd.read_csv(r5_path)
                raw_dfs.append(df_r5)
            else:
                print(f"  Warning: {r5_path} not found")
            
            # Load R_35 results (reps 5-39)
            r35_path = Path('results_R_35') / model / setting / 'raw_results.csv'
            if r35_path.exists():
                df_r35 = pd.read_csv(r35_path)
                # Adjust rep numbers: add 5 to each rep
                df_r35['rep'] = df_r35['rep'] + 5
                raw_dfs.append(df_r35)
            else:
                print(f"  Warning: {r35_path} not found")
            
            # Combine and save raw results
            if raw_dfs:
                combined_raw = pd.concat(raw_dfs, ignore_index=True)
                combined_raw = combined_raw.sort_values(['learner', 'tuner', 'rep'])
                raw_output_path = setting_output / 'raw_results.csv'
                combined_raw.to_csv(raw_output_path, index=False)
                print(f"  ✓ Saved {raw_output_path}")
                
                # Recalculate summary statistics
                summary_rows = []
                for (learner, tuner), group in combined_raw.groupby(['learner', 'tuner']):
                    summary_rows.append({
                        'learner': learner,
                        'tuner': tuner,
                        'pehe_mean': group['pehe'].mean(),
                        'pehe_std': group['pehe'].std(),
                        'pehe_plug_mean': group['pehe_plug'].mean(),
                        'pehe_plug_std': group['pehe_plug'].std(),
                        'n_reps': len(group)
                    })
                
                summary_df = pd.DataFrame(summary_rows)
                summary_output_path = setting_output / 'summary.csv'
                summary_df.to_csv(summary_output_path, index=False)
                print(f"  ✓ Saved {summary_output_path}")
            
            # Copy convergence folder (prefer R_35 if exists, otherwise R_5)
            convergence_copied = False
            for source_folder in ['results_R_35', 'results_R_5']:
                convergence_source = Path(source_folder) / model / setting / 'convergence'
                if convergence_source.exists():
                    convergence_output = setting_output / 'convergence'
                    if convergence_output.exists():
                        shutil.rmtree(convergence_output)
                    shutil.copytree(convergence_source, convergence_output)
                    print(f"  ✓ Copied convergence/ from {source_folder}")
                    convergence_copied = True
                    break
            
            if not convergence_copied:
                print(f"  Warning: No convergence folder found for {model}/{setting}")
            
            print()
    
    print("="*80)
    print("Combination complete! Results saved to results_R_40/")
    print("="*80)

if __name__ == "__main__":
    combine_results()