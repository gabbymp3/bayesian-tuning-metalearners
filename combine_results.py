#!/usr/bin/env python3
"""
Script to combine results from R_5, R_10, and R_20 folders into a new R_35 folder.
This creates 35 Monte Carlo repetitions for each model, tuner, and settings combination.
"""

import pandas as pd
import os
from pathlib import Path
import shutil

def combine_results():
    """Combine results from R_5, R_10, and R_20 into R_35."""
    
    # Define source folders - all now use the same naming convention
    source_folders = {
        'R_5': {
            'x_cb': ['1d', '2d', '4d', '6d'],
            'x_rf': ['1d', '2d', '4d', '6d']
        },
        'R_10': {
            'x_cb': ['1d', '2d', '4d', '6d'],
            'x_rf': ['1d', '2d', '4d', '6d']
        },
        'R_20': {
            'x_cb': ['1d', '2d', '4d', '6d'],
            'x_rf': ['1d', '2d', '4d', '6d']
        }
    }
    
    # Create output directory
    output_base = Path('results_R_35')
    output_base.mkdir(exist_ok=True)
    
    # Process each model (x_cb, x_rf)
    for model in ['x_cb', 'x_rf']:
        print(f"\nProcessing model: {model}")
        
        # Get the standard dimension folders for this model
        standard_dims = source_folders['R_5'][model]
        
        # Process each dimension
        for dim_folder in standard_dims:
            print(f"  Processing dimension: {dim_folder}")
            
            all_raw_data = []
            
            # Collect data from each source folder
            for source_name in ['R_5', 'R_10', 'R_20']:
                source_path = Path('results') / source_name / model
                raw_file = source_path / dim_folder / 'raw_results.csv'
                
                if raw_file.exists():
                    print(f"    Reading: {raw_file}")
                    df = pd.read_csv(raw_file)
                    
                    # Adjust rep numbers to be continuous
                    if source_name == 'R_10':
                        df['rep'] = df['rep'] + 5  # R_10 reps start at 5
                    elif source_name == 'R_20':
                        df['rep'] = df['rep'] + 15  # R_20 reps start at 15
                    # R_5 reps stay at 0-4
                    
                    all_raw_data.append(df)
                else:
                    print(f"    WARNING: File not found: {raw_file}")
            
            # Combine all raw data
            if all_raw_data:
                combined_raw = pd.concat(all_raw_data, ignore_index=True)
                combined_raw = combined_raw.sort_values(['learner', 'tuner', 'rep'])
                
                # Create output directory for this combination
                output_dir = output_base / model / dim_folder
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save combined raw results
                raw_output_file = output_dir / 'raw_results.csv'
                combined_raw.to_csv(raw_output_file, index=False)
                print(f"    Saved: {raw_output_file} ({len(combined_raw)} rows)")
                
                # Calculate and save summary statistics
                summary_data = []
                for (learner, tuner), group in combined_raw.groupby(['learner', 'tuner']):
                    summary_data.append({
                        'learner': learner,
                        'tuner': tuner,
                        'pehe_mean': group['pehe'].mean(),
                        'pehe_var': group['pehe'].var(),
                        'pehe_plug_mean': group['pehe_plug'].mean(),
                        'pehe_plug_var': group['pehe_plug'].var()
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_output_file = output_dir / 'summary.csv'
                summary_df.to_csv(summary_output_file, index=False)
                print(f"    Saved: {summary_output_file}")
    
    print(f"\n✓ All results combined successfully in {output_base}/")
    print(f"  Structure: results_R_35/{{x_cb,x_rf}}/{{1d,2d,4d,6d}}/{{raw_results.csv,summary.csv}}")

if __name__ == '__main__':
    combine_results()

