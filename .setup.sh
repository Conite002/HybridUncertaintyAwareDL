#!/bin/bash

echo "Starting project setup ..."

# Main Directories
mkdir -p Data
mkdir -p Data/Raw
mkdir -p Data/Preprocessed
mkdir -p Feature_Extraction
mkdir -p Uncertainty_Quantification
mkdir -p Conformal_Prediction
mkdir -p Explainability
mkdir -p Evaluation
mkdir -p Outputs
mkdir -p Outputs/logs
mkdir -p Outputs/models
mkdir -p Outputs/results
mkdir -p Experimentations
mkdir -p Scripts

# Organizing Multiple Datasets
mkdir -p Datasets
mkdir -p Datasets/SIPaKMeD
mkdir -p Datasets/Cytology_Cervical_Cancer
mkdir -p Datasets/Histopathology_Ovarian_Cancer
mkdir -p Datasets/Mammography_Breast_Cancer


# Subdirectories based on methodology
mkdir -p Uncertainty_Quantification/Single_Network
mkdir -p Uncertainty_Quantification/Monte_Carlo_Dropout
mkdir -p Uncertainty_Quantification/Deep_Ensemble
mkdir -p Conformal_Prediction/APS
mkdir -p Conformal_Prediction/RAPS
mkdir -p Conformal_Prediction/SAPS

# Important files
touch README.md
touch .gitignore
touch Scripts/train.sh
touch Scripts/evaluate.sh
touch Scripts/preprocess.sh
touch Scripts/run_experiments.sh

echo "Project setup complete."
