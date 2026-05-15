"""
Run script for CP1 - Data Processing and Baseline Modeling.

This script executes the full pipeline for CP1:
1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
3. Baseline Model Training
4. Multiple Model Experiments
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing import preprocess_pipeline
from src.modeling import modeling_pipeline
from notebooks.01_eda import main as run_eda


def main():
    print("="*70)
    print("CP1 PIPELINE - DATA PROCESSING AND MODELING")
    print("="*70)
    
    # Step 1: EDA
    print("\n" + "="*70)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("="*70)
    try:
        df_eda = run_eda()
        print("\n✓ EDA completed successfully")
    except Exception as e:
        print(f"\n✗ EDA failed: {str(e)}")
        print("Continuing with preprocessing...")
    
    # Step 2: Preprocessing
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    try:
        preprocessing_info = preprocess_pipeline()
        print("\n✓ Preprocessing completed successfully")
        print(f"  - Original shape: {preprocessing_info['original_shape']}")
        print(f"  - Final shape: {preprocessing_info['final_shape']}")
        print(f"  - Number of features: {preprocessing_info['n_features']}")
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {str(e)}")
        return
    
    # Step 3: Modeling
    print("\n" + "="*70)
    print("STEP 3: MODELING AND EXPERIMENTS")
    print("="*70)
    try:
        modeling_results = modeling_pipeline()
        print("\n✓ Modeling completed successfully")
        
        if 'experiment_df' in modeling_results and not modeling_results['experiment_df'].empty:
            print("\nTop 5 Models by RMSE:")
            print(modeling_results['experiment_df'].head(5)[['model', 'RMSE', 'MAPE (%)', 'R2']].to_string(index=False))
    except Exception as e:
        print(f"\n✗ Modeling failed: {str(e)}")
        return
    
    print("\n" + "="*70)
    print("CP1 PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nOutputs:")
    print("  - EDA plots: reports/images/")
    print("  - Processed data: data/processed/")
    print("  - Experiment results: reports/experiment_results.csv")
    print("  - Best model: models/best_model.pkl")


if __name__ == "__main__":
    main()
