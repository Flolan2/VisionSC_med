#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzes kinematic data using Linear Mixed Effects (LME) models for medication effects.

This script reads preprocessed kinematic data from CSV files located in a 'data'
subdirectory, performs outlier removal based on the 99th percentile, conducts
Principal Component Analysis (PCA) on kinematic variables, and fits LME models
to assess the effect of medication state (e.g., MED ON vs. MED OFF) on both
individual kinematic variables and principal components, while controlling for age.
Results, including PCA loadings and LME statistics (coefficients, p-values with
Bonferroni correction), are saved to CSV files in an 'Output' subdirectory.
"""

import pandas as pd
from statsmodels.formula.api import mixedlm
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def format_p_value(p):
    """Formats p-values for publication."""
    if p < 1e-10:
        return "<1e-10"
    elif p < 0.001:
        return "{:.2e}".format(p)
    else:
        return "{:.3f}".format(p)


def fit_mixedlm(formula, data, groups, max_retries=3):
    """
    Fits a MixedLM model with retries using different optimizers.

    Attempts to fit the model using 'lbfgs', 'cg', and 'bfgs' optimizers
    sequentially if convergence fails.

    Args:
        formula (str): The formula for the LME model (statsmodels format).
        data (pd.DataFrame): The dataframe containing the data.
        groups (pd.Series): The series indicating the grouping factor (e.g., Patient_ID).
        max_retries (int): The maximum number of different optimizers to try.

    Returns:
        statsmodels.regression.mixed_linear_model.MixedLMResultsWrapper or None:
        The fitted model result, or None if all attempts fail.
    """
    optimizers = ['lbfgs', 'cg', 'bfgs'] # List of optimizers to try in order
    for attempt in range(min(max_retries, len(optimizers))):
        try:
            model = mixedlm(formula, data, groups=groups)
            result = model.fit(method=optimizers[attempt])
            print(f"  LME fitting successful with optimizer '{optimizers[attempt]}'.")
            return result
        except Exception as e:
            print(f"  Attempt {attempt + 1}: LME fitting failed with optimizer '{optimizers[attempt]}'. Error: {e}")
            if attempt == max_retries - 1:
                print(f"  All optimization attempts failed for formula: {formula}")
                return None
            else:
                print("  Retrying with next optimizer...")
    return None # Should not be reached if max_retries > 0


def analyze_kinematic_data(input_file_path, condition_var, condition_name, output_csv_filename):
    """
    Performs PCA and LME analysis on kinematic data for a specific condition.

    Args:
        input_file_path (str): Path to the input CSV file (e.g., data/summary_ft.csv).
        condition_var (str): Column name for the condition variable (e.g., 'Condition_MED').
        condition_name (str): Descriptive name for the condition analysis (e.g., 'Finger Tapping MED').
        output_csv_filename (str): Base filename for the output LME results CSV.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, 'output')
    output_csv_path = os.path.join(output_folder, output_csv_filename)
    # Ensure condition_name is filesystem-safe for filenames
    safe_condition_name = condition_name.replace(" ", "_").replace("/", "-")
    output_pca_loadings_path = os.path.join(output_folder, f'PCA_Loadings_{safe_condition_name}.csv')
    output_pca_results_path = os.path.join(output_folder, f'PCA_LME_Results_{safe_condition_name}.csv')

    os.makedirs(output_folder, exist_ok=True)
    print(f"\n--- Processing Condition: {condition_name} ---")
    print(f"Input File: {input_file_path}")
    print(f"Condition Variable: {condition_var}")

    # --- Data Loading and Preprocessing ---
    try:
        df = pd.read_csv(input_file_path, delimiter=';', decimal='.')
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_file_path}. Skipping analysis.")
        return
    except Exception as e:
        print(f"ERROR: Failed to read input file {input_file_path}. Error: {e}. Skipping analysis.")
        return


    df.columns = [col.replace(' ', '_') for col in df.columns]

    # Ensure required columns exist
    required_base_cols = ['Patient_ID', 'Birthdate', 'Date_of_Visit', condition_var]
    if not all(col in df.columns for col in required_base_cols):
        missing_cols = [col for col in required_base_cols if col not in df.columns]
        print(f"ERROR: Missing required columns: {missing_cols} in file {input_file_path}. Skipping analysis for {condition_name}.")
        return

    # Handle condition variable missing values and type
    # Convert condition to string first to handle potential mixed types before replacing
    df[condition_var] = df[condition_var].astype(str).replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
    # Attempt conversion to category after cleaning
    try:
        df[condition_var] = df[condition_var].astype('category')
    except TypeError:
        print(f"Warning: Could not convert {condition_var} to category. Contains non-string/NA values after cleaning? Proceeding with object type.")


    # Calculate Age at Visit
    df['Birthdate'] = pd.to_datetime(df['Birthdate'], dayfirst=True, errors='coerce')
    df['Date_of_Visit'] = pd.to_datetime(df['Date_of_Visit'], dayfirst=True, errors='coerce')
    # Check if dates were parsed correctly before calculating age
    if df['Birthdate'].isnull().any() or df['Date_of_Visit'].isnull().any():
        print("Warning: Some 'Birthdate' or 'Date_of_Visit' values could not be parsed. Resulting 'Age_at_Visit' may have NaNs.")
    df['Age_at_Visit'] = (df['Date_of_Visit'] - df['Birthdate']).dt.days / 365.25

    # Define kinematic variables (ensure they exist in the dataframe)
    potential_kinematic_vars = [
        'meanamplitude', 'stdamplitude', 'meanrmsvelocity', 'stdrmsvelocity',
        'meanopeningspeed', 'stdopeningspeed', 'meanclosingspeed',
        'stdclosingspeed', 'meancycleduration', 'stdcycleduration',
        'rangecycleduration', 'amplitudedecay', 'velocitydecay', 'ratedecay',
        'cvamplitude', 'cvcycleduration', 'rate', 'meanspeed', 'stdspeed', 'cvspeed'
    ]
    kinematic_vars = [var for var in potential_kinematic_vars if var in df.columns]
    if not kinematic_vars:
        print("ERROR: No kinematic variables found in the dataframe columns. Skipping analysis.")
        return
    print(f"Using Kinematic Variables: {kinematic_vars}")

    # --- Outlier Removal (based on 99th percentile) ---
    # Ensure calculation is done only on numeric data
    numeric_kinematic_df = df[kinematic_vars].apply(pd.to_numeric, errors='coerce')
    percentiles_99 = numeric_kinematic_df.quantile(0.99)

    # --- PCA Analysis ---
    print("\n--- Starting PCA Analysis ---")
    pca_lme_results = []

    # Filter data for PCA (requires non-missing values for all kinematic vars, ID, condition, age)
    cols_for_pca_filter = ['Patient_ID'] + kinematic_vars + [condition_var, 'Age_at_Visit']
    df_filtered_pca = df.dropna(subset=cols_for_pca_filter).copy() # Use .copy() to avoid SettingWithCopyWarning

    # Apply 99th percentile filter (ensure numeric comparison)
    for var in kinematic_vars:
         # Convert column to numeric for comparison, coercing errors
         df_filtered_pca[var] = pd.to_numeric(df_filtered_pca[var], errors='coerce')
         # Drop rows where conversion failed before filtering
         df_filtered_pca = df_filtered_pca.dropna(subset=[var])
         # Apply percentile filter
         if var in percentiles_99 and not pd.isna(percentiles_99[var]):
             df_filtered_pca = df_filtered_pca[df_filtered_pca[var] <= percentiles_99[var]]
         else:
              print(f"  Warning: Skipping 99th percentile filter for {var} in PCA (percentile is NaN or missing).")


    print(f"Data points available for PCA after filtering: {df_filtered_pca.shape[0]}")

    # Need enough samples relative to features for PCA/LME
    if df_filtered_pca.shape[0] < max(3, len(kinematic_vars) + 1):
        print("Insufficient data for PCA after filtering (too few samples or too many NaNs after conversion/filtering). Skipping PCA analysis.")
    else:
        # Map condition variable to standardized 'OFF'/'ON' levels
        condition_mapping = {
            # Focus on MED conditions, map common variations
            'OFF': 'OFF', 'ON': 'ON',
            'OFF MEDICATION': 'OFF', 'ON MEDICATION': 'ON',
            'OFF MED': 'OFF', 'ON MED': 'ON',
            'OFFMED': 'OFF', 'ONMED': 'ON', # Handle no space cases
            # Map potential DBS variations to handle mixed data if present, though DBS condition isn't explicitly analyzed
            'OFF DBS': 'OFF', 'ON DBS': 'ON',
            'OFFDBS': 'OFF', 'ONDBS': 'ON'
        }
        # Ensure the column is string before mapping
        df_filtered_pca[condition_var] = df_filtered_pca[condition_var].astype(str).str.upper().str.strip() # Add strip
        df_filtered_pca[condition_var] = df_filtered_pca[condition_var].map(condition_mapping)
        df_filtered_pca = df_filtered_pca.dropna(subset=[condition_var]) # Drop rows where mapping failed (unrecognized values)

        if df_filtered_pca.shape[0] < 3:
             print("Insufficient data after condition mapping (less than 3 valid samples). Skipping PCA analysis.")
        else:
            df_filtered_pca[condition_var] = df_filtered_pca[condition_var].astype('category')
            # Set categories explicitly to ensure 'OFF' is baseline
            df_filtered_pca[condition_var] = df_filtered_pca[condition_var].cat.set_categories(['OFF', 'ON'], ordered=True)

            # *** CORRECTED CHECK HERE ***
            # Check if both levels 'OFF' and 'ON' are actually present in the filtered data
            num_present_levels_pca = len(df_filtered_pca[condition_var].dropna().unique())
            if num_present_levels_pca < 2:
                 print(f"  Only {num_present_levels_pca} condition level(s) present after filtering/mapping for PCA. Skipping PCA LME analysis.")
                 # Skip the rest of the PCA LME block by not entering the else

            else: # *** START OF PCA LME BLOCK (if enough levels present) ***
                # Standardize kinematic variables for PCA
                scaler = StandardScaler()
                # Ensure data is numeric before scaling
                kinematic_data_numeric = df_filtered_pca[kinematic_vars].apply(pd.to_numeric, errors='coerce')
                # Check for NaNs introduced by coercion before scaling
                if kinematic_data_numeric.isnull().any().any():
                     print("Warning: NaNs found in kinematic data before scaling. Rows with NaNs will be implicitly dropped by PCA.")

                # Fit scaler and transform on non-NaN data only
                valid_indices = kinematic_data_numeric.dropna().index
                if len(valid_indices) < max(3, len(kinematic_vars) + 1):
                     print("Insufficient non-NaN data points remaining for PCA scaling. Skipping PCA analysis.")
                else:
                    kinematic_data_scaled = scaler.fit_transform(kinematic_data_numeric.loc[valid_indices])
                    df_pca_analysis_base = df_filtered_pca.loc[valid_indices].copy() # Select only valid rows for analysis


                    # Perform PCA
                    pca = PCA()
                    principal_components = pca.fit_transform(kinematic_data_scaled)
                    print(f"PCA performed on {principal_components.shape[0]} samples.")

                    # Create DataFrame with PCA scores
                    pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
                    df_pca_scores = pd.DataFrame(data=principal_components, columns=pca_columns, index=valid_indices) # Use valid indices
                    # Concatenate using the index
                    df_pca_analysis = pd.concat([df_pca_analysis_base[['Patient_ID', condition_var, 'Age_at_Visit']], df_pca_scores], axis=1)
                    # Drop rows where concatenation might have failed or crucial columns are NaN
                    df_pca_analysis.dropna(subset=pca_columns + ['Patient_ID', condition_var, 'Age_at_Visit'], inplace=True)


                    # Save PCA Loadings
                    loadings = pd.DataFrame(pca.components_.T, columns=pca_columns, index=kinematic_vars)
                    loadings.to_csv(output_pca_loadings_path)
                    print(f"PCA loadings saved to {output_pca_loadings_path}")

                    # Analyze top PCA scores using LME
                    num_components_to_analyze = min(5, len(pca_columns))
                    print(f"Analyzing top {num_components_to_analyze} principal components with LME...")

                    for pc in pca_columns[:num_components_to_analyze]:
                        print(f"Fitting LME for {pc}...")
                        # Check again for sufficient data after potential drops
                        if df_pca_analysis.shape[0] < 3 or len(df_pca_analysis[condition_var].unique()) < 2:
                            print(f"  Insufficient data points or condition levels remain for LME on {pc}. Skipping.")
                            continue

                        formula = f"{pc} ~ C({condition_var}, Treatment(reference='OFF')) + Age_at_Visit" # Explicit reference
                        result = fit_mixedlm(formula, df_pca_analysis, groups=df_pca_analysis["Patient_ID"])

                        if result:
                            # Extract relevant coefficients (only for the condition effect, level 'ON')
                            param_name = f"C({condition_var}, Treatment(reference='OFF'))[T.ON]"
                            if param_name in result.params:
                                coeff = result.params[param_name]
                                pval = result.pvalues[param_name]
                                # Percentage change isn't meaningful for standardized PCs
                                percentage_change = np.nan

                                pca_lme_results.append({
                                    'Principal_Component': pc,
                                    f'{condition_name}_Level': 'ON', # Comparing ON vs OFF (baseline)
                                    f'{condition_name}_Coefficient': round(coeff, 3),
                                    f'{condition_name}_P_Value': pval,
                                    f'{condition_name}_Percentage_Change': percentage_change,
                                    'Data_Points_Analyzed': df_pca_analysis.shape[0] # Use shape after potential drops
                                })
                            else:
                                 print(f"  Could not find parameter '{param_name}' in LME results for {pc}. Available params: {list(result.params.keys())}")
                        else:
                            print(f"  LME fitting failed for {pc}.")

                    # Save PCA LME results with Bonferroni correction
                    if pca_lme_results:
                        pca_results_df = pd.DataFrame(pca_lme_results)
                        n_tests_pca = len(pca_results_df)
                        bonferroni_alpha_pca = 0.05 / n_tests_pca if n_tests_pca > 0 else np.nan

                        pca_results_df['Significant_After_Bonferroni'] = pca_results_df[f'{condition_name}_P_Value'] < bonferroni_alpha_pca
                        pca_results_df['Significant_After_Bonferroni'] = pca_results_df['Significant_After_Bonferroni'].map({True: 'Yes', False: 'No'})
                        pca_results_df[f'{condition_name}_P_Value_Formatted'] = pca_results_df[f'{condition_name}_P_Value'].apply(format_p_value) # Keep original, add formatted
                        pca_results_df['Bonferroni_Corrected_Alpha'] = bonferroni_alpha_pca

                        # Reorder columns for clarity
                        pca_cols_order = ['Principal_Component', f'{condition_name}_Level', f'{condition_name}_Coefficient',
                                          f'{condition_name}_P_Value', f'{condition_name}_P_Value_Formatted',
                                          'Bonferroni_Corrected_Alpha', 'Significant_After_Bonferroni',
                                          f'{condition_name}_Percentage_Change', 'Data_Points_Analyzed']
                        # Ensure all columns exist before reordering
                        pca_cols_order = [col for col in pca_cols_order if col in pca_results_df.columns]
                        pca_results_df = pca_results_df[pca_cols_order]


                        pca_results_df.to_csv(output_pca_results_path, index=False)
                        print(f"PCA LME results saved to {output_pca_results_path} ({n_tests_pca} tests conducted)")
                    else:
                        print("No PCA LME results to save.")
            # *** END OF PCA LME BLOCK ***

    # --- Individual Kinematic Variable LME Analysis ---
    # (This part remains unchanged as the error was in the PCA block)
    print("\n--- Starting Individual Variable LME Analysis ---")
    individual_lme_results = []

    for var in kinematic_vars:
        print(f"Processing variable: {var}")

        # Filter data for the specific variable
        df_filtered_var = df.dropna(subset=['Patient_ID', var, condition_var, 'Age_at_Visit']).copy()

        # Apply 99th percentile filter (ensure numeric comparison)
        df_filtered_var[var] = pd.to_numeric(df_filtered_var[var], errors='coerce')
        df_filtered_var = df_filtered_var.dropna(subset=[var]) # Drop if conversion failed
        if var in percentiles_99 and not pd.isna(percentiles_99[var]): # Check if percentile exists and is valid
             df_filtered_var = df_filtered_var[df_filtered_var[var] <= percentiles_99[var]]
        else:
             print(f"  Warning: Could not apply 99th percentile filter for {var} (percentile is NaN or missing).")


        n_points_var = df_filtered_var.shape[0]
        # print(f"  Data points available for {var} after filtering: {n_points_var}") # Less verbose

        if n_points_var < 3:
            print(f"  Insufficient data points for {var} after initial filtering. Skipping.")
            continue

        # Map condition variable to standardized 'OFF'/'ON' levels
        df_filtered_var[condition_var] = df_filtered_var[condition_var].astype(str).str.upper().str.strip() # Add strip
        df_filtered_var[condition_var] = df_filtered_var[condition_var].map(condition_mapping)
        df_filtered_var = df_filtered_var.dropna(subset=[condition_var]) # Drop if mapping failed


        if df_filtered_var.empty or df_filtered_var.shape[0] < 3 :
            print(f"  No data or insufficient data remaining after condition mapping for {var}. Skipping.")
            continue

        df_filtered_var[condition_var] = df_filtered_var[condition_var].astype('category')
        # Set categories explicitly to ensure 'OFF' is baseline
        df_filtered_var[condition_var] = df_filtered_var[condition_var].cat.set_categories(['OFF', 'ON'], ordered=True)

        # Check if both levels exist after filtering for this specific variable
        num_present_levels_var = len(df_filtered_var[condition_var].dropna().unique())
        if num_present_levels_var < 2:
            present_levels = df_filtered_var[condition_var].dropna().unique().tolist()
            print(f"  Only {num_present_levels_var} condition level(s) ({present_levels}) present for {var} after filtering. Skipping LME.")
            continue


        # Calculate baseline mean (OFF state) for percentage change calculation
        baseline_data = df_filtered_var[df_filtered_var[condition_var] == 'OFF']
        if baseline_data.empty or baseline_data[var].isnull().all():
            # print(f"  No valid baseline (OFF) data found for {var}. Percentage change will be NaN.") # Less verbose
            baseline_mean = np.nan
        else:
            baseline_mean = baseline_data[var].mean() # Mean of non-NaN values

        # Fit LME model
        formula = f"{var} ~ C({condition_var}, Treatment(reference='OFF')) + Age_at_Visit" # Explicit reference
        result = fit_mixedlm(formula, df_filtered_var, groups=df_filtered_var["Patient_ID"])

        if result:
            # Extract relevant coefficients (only for the condition effect, level 'ON')
            param_name = f"C({condition_var}, Treatment(reference='OFF'))[T.ON]"
            if param_name in result.params:
                coeff = result.params[param_name]
                pval = result.pvalues[param_name]

                # Calculate percentage change relative to baseline mean
                if baseline_mean != 0 and not np.isnan(baseline_mean):
                    percentage_change = (coeff / baseline_mean) * 100
                else:
                    percentage_change = np.nan # Avoid division by zero or if baseline is missing/zero

                individual_lme_results.append({
                    'Variable': var,
                    f'{condition_name}_Level': 'ON', # Comparing ON vs OFF (baseline)
                    f'{condition_name}_Coefficient': round(coeff, 3),
                    f'{condition_name}_P_Value': pval,
                    f'{condition_name}_Percentage_Change': round(percentage_change, 2) if not np.isnan(percentage_change) else np.nan,
                    'Data_Points_Analyzed': n_points_var # Initial count before LME might drop more NaNs internally
                })
            else:
                 print(f"  Could not find parameter '{param_name}' in LME results for {var}. Available params: {list(result.params.keys())}")
        else:
            print(f"  LME fitting failed for {var}.")

    # Save individual LME results with Bonferroni correction
    if individual_lme_results:
        individual_results_df = pd.DataFrame(individual_lme_results)
        n_tests_indiv = len(individual_results_df)
        bonferroni_alpha_indiv = 0.05 / n_tests_indiv if n_tests_indiv > 0 else np.nan

        individual_results_df['Significant_After_Bonferroni'] = individual_results_df[f'{condition_name}_P_Value'] < bonferroni_alpha_indiv
        individual_results_df['Significant_After_Bonferroni'] = individual_results_df['Significant_After_Bonferroni'].map({True: 'Yes', False: 'No'})
        individual_results_df[f'{condition_name}_P_Value_Formatted'] = individual_results_df[f'{condition_name}_P_Value'].apply(format_p_value) # Keep original, add formatted
        individual_results_df['Bonferroni_Corrected_Alpha'] = bonferroni_alpha_indiv

        # Reorder columns for clarity
        indiv_cols_order = ['Variable', f'{condition_name}_Level', f'{condition_name}_Coefficient',
                            f'{condition_name}_Percentage_Change', f'{condition_name}_P_Value',
                            f'{condition_name}_P_Value_Formatted', 'Bonferroni_Corrected_Alpha',
                            'Significant_After_Bonferroni', 'Data_Points_Analyzed']
        # Ensure all columns exist before reordering
        indiv_cols_order = [col for col in indiv_cols_order if col in individual_results_df.columns]
        individual_results_df = individual_results_df[indiv_cols_order]


        individual_results_df.to_csv(output_csv_path, index=False)
        print(f"\nIndividual variable LME results saved to {output_csv_path} ({n_tests_indiv} tests conducted)")
    else:
        print("\nNo individual variable LME results to save.")

    print(f"--- Finished Processing: {condition_name} ---")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the subdirectory containing the data files
    data_dir = os.path.join(script_dir, 'data')

    # Define the analysis tasks: Task Name -> Input Filename within data_dir
    # Using more descriptive keys now aligning with filenames
    tasks = {
        'Finger_Tapping': 'summary_ft.csv',
        'Hand_Movement': 'summary_hm.csv' # Changed key to Hand_Movement for clarity
        # Add more tasks here if needed, mapping a descriptive name to its filename
    }

    print("Starting Medication Effect Analyses...")
    print(f"Looking for input files in: {data_dir}")

    analysis_performed = False
    for task_key, input_filename in tasks.items():
        # Construct the full path to the input file
        input_file_path = os.path.join(data_dir, input_filename)

        # Check if the specific input file exists before attempting analysis
        if not os.path.exists(input_file_path):
            print(f"\nWARNING: Input file not found for task '{task_key}': {input_file_path}. Skipping this task.")
            continue

        analysis_performed = True # Mark that at least one analysis will be attempted
        task_name_formatted = task_key.replace("_", " ") # For display names

        # --- Analyze MED Condition Only ---
        analyze_kinematic_data(
            input_file_path=input_file_path, # Use the constructed path
            condition_var='Condition_MED', # Use the medication condition column
            condition_name=f"{task_name_formatted} MED", # Name for output files and logs
            output_csv_filename=f'LMM_Results_{task_key}_MED.csv' # Output filename based on task key
        )

    if not analysis_performed:
         print("\nNo input files were found in the 'data' directory. No analyses were run.")
    else:
        print("\nAll medication analyses complete.")