import os
import json
import pandas as pd

this_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(this_dir, "data")
SCRIPT_DIR = os.path.join(this_dir, "scripts")

# Folder data paths
DATA_RAW_PATH = os.path.join(DATA_DIR, "raw")
DATA_CLEAN_PATH = os.path.join(DATA_DIR, "clean")
DATA_RESULTS_PATH = os.path.join(DATA_DIR, "results")

# raw data
RESISTANCE_RAW=os.path.join(DATA_RAW_PATH , "resistance_raw")
Top5_Similarity_Summary=os.path.join(RESISTANCE_RAW , "Top5_Similarity_Summary.csv")
Top8_Similarity_Summary=os.path.join(RESISTANCE_RAW , "Top8_Similarity_Summary.csv")
EDX_min_max_summary=os.path.join(RESISTANCE_RAW , "EDX_min_max_summary.csv")
MAPPED_CENTROIDS_JSON = os.path.join(RESISTANCE_RAW , "centroids_mapped_with_indices.json")
MAPPED_CENTROIDS_JSON_8 = os.path.join(RESISTANCE_RAW , "centroids_mapped_with_indices_8.json")

# images paths
Au_Pd_Pt_Rh_0010304=os.path.join(RESISTANCE_RAW, "0010304")
Au_Pd_Pt_Rh_photo_0010304=os.path.join(Au_Pd_Pt_Rh_0010304 , "0010304_Au-Pd-Pt-Rh_photo.jpg")

Ag_Au_Cu_Pd_Pt_0010403=os.path.join(RESISTANCE_RAW, "0010403")
Ag_Au_Cu_Pd_Pt_photo_0010403=os.path.join(Ag_Au_Cu_Pd_Pt_0010403 , "Photo_0010403_Ag-Au-Cu-Pd-Pt_asdepo.jpg")

Au_Pd_Pt_Rh_Ru_0010311=os.path.join(RESISTANCE_RAW, "0010311")
Au_Pd_Pt_Rh_Ru_photo_0010311=os.path.join(Au_Pd_Pt_Rh_Ru_0010311 , "0010311_Au-Pd-Pt-Rh-Ru_photo.jpg")

Ag_Au_Pd_Pt_Rh_RT_0010275=os.path.join(RESISTANCE_RAW, "0010275")
Ag_Au_Pd_Pt_Rh_RT_photo_0010275=os.path.join(Ag_Au_Pd_Pt_Rh_RT_0010275 , "0010275_Ag-Au-Pd-Pt-Rh_on_15nm_Ta_photo.jpg")

Ag_Au_Pd_0010272=os.path.join(RESISTANCE_RAW, "0010272")
Ag_Au_Pd_photo_0010272=os.path.join(Ag_Au_Pd_0010272 , "0010272_Ag-Au-Pd_on_15nm_Ta_photo.jpg")

Ir_Pd_Pt_Rh_Ru_0010374=os.path.join(RESISTANCE_RAW, "0010374")
Ir_Pd_Pt_Rh_Ru_photo_0010374=os.path.join(Ir_Pd_Pt_Rh_Ru_0010374 , "Photo_0010374_Ir-Pd-Pt-Rh-Ru_as_depo.jpg")

Ag_Au_Pd_Pt_0010402=os.path.join(RESISTANCE_RAW, "0010402")
Ag_Au_Pd_Pt_photo_0010402=os.path.join(Ag_Au_Pd_Pt_0010402 , "Photo_0010402_Ag-Au-Pd-Pt_asdepo.JPG")

Au_Cu_Pd_Pt_0010399=os.path.join(RESISTANCE_RAW, "0010399")
Au_Cu_Pd_Pt_photo_0010399=os.path.join(Au_Cu_Pd_Pt_0010399 , "Photo_0010399_Au-Cu-Pd-Pt_asdepo.jpg")

# Clean data
DATA_CLEAN_InIT_CHOICES = os.path.join(DATA_CLEAN_PATH, "init_choices")
INIT_CHOICES_N8_DIR = os.path.join(DATA_CLEAN_PATH, "DATA_CLEAN_InIT_CHOICES_8")

RESISTANCE_CLEANED_FILES = os.path.join(DATA_CLEAN_PATH, "resistance_cleaned_files")
DATASET_10272_Ag_Au_Pd_RT= os.path.join(RESISTANCE_CLEANED_FILES, "10272_Ag-Au-Pd_RT.csv")
DATASET_10275_Ag_Au_Pd_Pt_Rh_RT= os.path.join(RESISTANCE_CLEANED_FILES, "10275_Ag-Au-Pd-Pt-Rh_RT.csv")
DATASET_10304_Au_Pd_Pt_Rh_RT= os.path.join(RESISTANCE_CLEANED_FILES, "10304_Au-Pd-Pt-Rh_RT.csv")
DATASET_10311_Au_Pd_Pt_Rh_Ru_RT= os.path.join(RESISTANCE_CLEANED_FILES, "10311_Au-Pd-Pt-Rh-Ru_RT.csv")
DATASET_10403_Ag_Au_Cu_Pd_Pt_RT= os.path.join(RESISTANCE_CLEANED_FILES, "10403_Ag-Au-Cu-Pd-Pt_RT.csv")
DATASET_10402_Ag_Au_Pd_Pt_RT= os.path.join(RESISTANCE_CLEANED_FILES, "10402_Ag-Au-Pd-Pt_RT.csv")
DATASET_10399_Au_Cu_Pd_Pt_RT= os.path.join(RESISTANCE_CLEANED_FILES, "10399_Au-Cu-Pd-Pt_RT.csv")
DATASET_10374_Ir_Pd_Pt_Rh_Ru = os.path.join(RESISTANCE_CLEANED_FILES, "10374_Ir_Pd_Pt-Rh_Ru.csv")


# Results data
Wafer_Output_Dir= os.path.join(DATA_RESULTS_PATH , "wafer_output_black")
MiX_SELECTION_ANALYSIS = os.path.join(DATA_RESULTS_PATH , "mix_selection_analysis")

UNCERTAINTY_PATH = os.path.join(DATA_RESULTS_PATH, "Uncertainty")

Results_10272 = os.path.join(UNCERTAINTY_PATH, "10272_results")
MAE_PRIORS_10272 = os.path.join(Results_10272, "mae_priors_results.csv")

Results_10275 = os.path.join(UNCERTAINTY_PATH, "10275_results")
MAE_PRIORS_10275 = os.path.join(Results_10275, "mae_priors_results.csv")

Results_10304 = os.path.join(UNCERTAINTY_PATH, "10304_results")
MAE_PRIORS_10304 = os.path.join(Results_10304, "mae_priors_results.csv")

Results_10311 = os.path.join(UNCERTAINTY_PATH, "10311_results")
MAE_PRIORS_10311 = os.path.join(Results_10311, "mae_priors_results.csv")

Results_10374 = os.path.join(UNCERTAINTY_PATH, "10374_results")
MAE_PRIORS_10374 = os.path.join(Results_10374, "mae_priors_results.csv")

Results_10399 = os.path.join(UNCERTAINTY_PATH, "10399_results")
MAE_PRIORS_10399 = os.path.join(Results_10399, "mae_priors_results.csv")

Results_10402 = os.path.join(UNCERTAINTY_PATH, "10402_results")
MAE_PRIORS_10402 = os.path.join(Results_10402, "mae_priors_results.csv")

Results_10403 = os.path.join(UNCERTAINTY_PATH, "10403_results")
MAE_PRIORS_10403 = os.path.join(Results_10403, "mae_priors_results.csv")

#SEEDS of 10311 dataset
UNCERTAINTY_PATH_SEEDS = os.path.join(UNCERTAINTY_PATH, "SEEDS")
Results_10311_uncertainty_seeds = os.path.join(UNCERTAINTY_PATH_SEEDS, "10311_results")

# Results data for UNCERTAINTY_PATH_10
UNCERTAINTY_PATH_8 = os.path.join(DATA_RESULTS_PATH, "Uncertainty_8_points")


# Sawei results
SAWEI_PATH = os.path.join(DATA_RESULTS_PATH, "Sawei")

Results_10374_sawei = os.path.join(SAWEI_PATH, "10374_results")
MAE_PRIORS_10374_sawei = os.path.join(Results_10374_sawei, "mae_priors_results.csv")

Results_10399_sawei = os.path.join(SAWEI_PATH, "10399_results")
MAE_PRIORS_10399_sawei = os.path.join(Results_10399_sawei, "mae_priors_results.csv")

Results_10402_sawei = os.path.join(SAWEI_PATH, "10402_results")
MAE_PRIORS_10402_sawei = os.path.join(Results_10402_sawei, "mae_priors_results.csv")

Results_10403_sawei = os.path.join(SAWEI_PATH, "10403_results")
MAE_PRIORS_10403_sawei = os.path.join(Results_10403_sawei, "mae_priors_results.csv")

Results_10311_sawei = os.path.join(SAWEI_PATH, "10311_results")
MAE_PRIORS_10311_sawei = os.path.join(Results_10311_sawei, "mae_priors_results.csv")

Results_10304_sawei = os.path.join(SAWEI_PATH, "10304_results")
MAE_PRIORS_10304_sawei = os.path.join(Results_10304_sawei, "mae_priors_results.csv")

Results_10275_sawei = os.path.join(SAWEI_PATH, "10275_results")
MAE_PRIORS_10275_sawei = os.path.join(Results_10275_sawei, "mae_priors_results.csv")

Results_10272_sawei = os.path.join(SAWEI_PATH, "10272_results")
MAE_PRIORS_10272_sawei = os.path.join(Results_10272_sawei, "mae_priors_results.csv")

# evenly_spaced_points
EVENLY_PATH= os.path.join(DATA_RESULTS_PATH, "evenly_spaced_points")
EVENLY_PATH_Uncertainty= os.path.join(EVENLY_PATH, "Uncertainty")
EVENLY_PATH_Sawei= os.path.join(EVENLY_PATH, "Sawei")

#SEEDS of 10311 dataset
SAWEI_PATH_SEEDS = os.path.join(SAWEI_PATH, "SEEDS")
Results_10311_sawei_seeds = os.path.join(SAWEI_PATH_SEEDS, "10311_results")
# Sawei results_8
SAWEI_PATH_8 = os.path.join(DATA_RESULTS_PATH, "Sawei_8_points")

# Compare results
Compare_Result = os.path.join(DATA_RESULTS_PATH, "Compare_result")
summary_folder_Uncertainty = os.path.join(Compare_Result, "results_summary_uncertainty")
summary_folder_SAWEI = os.path.join(Compare_Result, "results_summary_sawei")
evenl_mixed_compare_Uncertainty = os.path.join(Compare_Result, "evenl_mixed_compare_Uncertainty")
evenl_mixed_compare_SAWEI = os.path.join(Compare_Result, "evenl_mixed_compare_SAWEI")
base_mixed_Uncertainty = os.path.join(Compare_Result, "base_mixed_Uncertainty")
base_mixed_SAWEI= os.path.join(Compare_Result, "base_mixed_SAWEI")
MAE_PLOT = os.path.join(DATA_RESULTS_PATH, "MAE_plot")



