import torch
import torch.nn as nn
import torch.optim as optim         
import os, sys
import numpy as np
import argparse
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
from train import train_model, train_deep_ensemble, get_dataloader
from test import evaluate_model, evaluate_deep_ensemble, evaluate_mc_dropout
from models import SingleNetwork, MCDropoutNetwork
from utils.utils import enable_dropout
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from uncertainty_analysis import rejection_plot
from calibration import expected_calibration_error, reliability_diagram

import json, logging
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_ROOT)

# fixed random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(42)


BATCH_SIZES = [64, 128] 
EPOCHS = 30
ENSEMBLE_SIZE = 5
LEARNING_RATE = 1e-4
DROP_PROBS = [0.6] 
DEVICE = get_device()
FEATURES_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "Outputs/models")
RESULTS_CSV_PATH = os.path.join(PROJECT_ROOT, "Outputs/results/hyperparameter_results.csv")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
results = []
total_combinations = len(BATCH_SIZES) * len(DROP_PROBS)

import itertools
from tqdm import tqdm
import pandas as pd



# 📌 Main Training Pipeline
if __name__ == "__main__":
    
    
    with tqdm(total=total_combinations, desc="Hyperparameter Tuning Progress") as pbar:
        for BATCH_SIZE, DROPOUT_RATE in itertools.product(BATCH_SIZES, DROP_PROBS):

            logging.info(f"🚀 Training with Batch={BATCH_SIZE}, LR={LEARNING_RATE}, Dropout={DROPOUT_RATE}...")

            # ✅ Load Data
            train_loader = get_dataloader("train", BATCH_SIZE)
            val_loader = get_dataloader("val", BATCH_SIZE)
            test_loader = get_dataloader("test", BATCH_SIZE)

            # ========================
            # 📌 Train Single Network
            # ========================
            logging.info("Training Single Network...")
            single_net = SingleNetwork(dropout_rate=DROPOUT_RATE).to(DEVICE)
            optimizer = optim.Adam(single_net.parameters(), lr=LEARNING_RATE)
            criterion = nn.CrossEntropyLoss()

            train_model(single_net, train_loader, val_loader, criterion, optimizer, EPOCHS, LEARNING_RATE, "SingleNetwork", patience=15)
            acc_sn, prec_sn, rec_sn, f1_sn, auc_sn = evaluate_model(single_net, test_loader, "SingleNetwork")
            logging.info("✅ Single Network Training Complete!")

            # ✅ Store results in dictionary
            results.append({
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "dropout_rate": DROPOUT_RATE,
                "accuracy_single": acc_sn,
                "precision_single": prec_sn,
                "recall_single": rec_sn,
                "f1_single": f1_sn,
                "auc_single": auc_sn,
            })

            # ✅ Save results to CSV file after each iteration
            df_results = pd.DataFrame(results)
            df_results.to_csv(RESULTS_CSV_PATH, index=False)

            # ✅ Update progress bar
            pbar.update(1)

        logging.info(f"✅ Hyperparameter tuning complete! Results saved in {RESULTS_CSV_PATH}")

    
    
    
    
    
    
    
    
    
    
    
    # for BATCH_SIZE, LEARNING_RATE, DROPOUT_RATE in itertools.product(BATCH_SIZES, LEARNING_RATES, DROP_PROBS):
    #     logging.info(f"🚀 Training with Batch Size={BATCH_SIZE}, Epochs={EPOCHS}, LR={LEARNING_RATE}, Dropout={DROPOUT_RATE}...")

    #     logging.info("Loading Data...")
    #     train_loader = get_dataloader("train", BATCH_SIZE)
    #     val_loader = get_dataloader("val", BATCH_SIZE)
    #     test_loader = get_dataloader("test", BATCH_SIZE)

    #     # 📌 Train Single Network
    #     logging.info("Training Single Network...")
    #     single_net = SingleNetwork(dropout_rate=DROPOUT_RATE).to(DEVICE)
    #     optimizer = optim.Adam(single_net.parameters(), lr=LEARNING_RATE)
    #     criterion = nn.CrossEntropyLoss()
        
    #     train_model(single_net, train_loader, val_loader, criterion, optimizer, EPOCHS, LEARNING_RATE, "SingleNetwork", patience=10)
    #     evaluate_model(single_net, test_loader, "SingleNetwork")
    #     torch.save(single_net.state_dict(), os.path.join(MODEL_SAVE_PATH, "single_network.pth"))
    #     logging.info("Single Network Training Complete!")

        # # 📌 Train Monte Carlo Dropout Network
        # logging.info("Training Monte Carlo Dropout Network...")
        # mc_dropout_net = MCDropoutNetwork().to(DEVICE)
        # optimizer = optim.Adam(mc_dropout_net.parameters(), lr=LEARNING_RATE)
        # enable_dropout(mc_dropout_net)
        # train_model(mc_dropout_net, train_loader, val_loader, criterion, optimizer, EPOCHS, LEARNING_RATE, "MCDropout")
        # torch.save(mc_dropout_net.state_dict(), os.path.join(MODEL_SAVE_PATH, "mc_dropout.pth"))
        # logging.info("Monte Carlo Dropout Training Complete!")


        # logging.info("Training Deep Ensemble...")
        # train_deep_ensemble(train_loader, val_loader, ensemble_size=ENSEMBLE_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE)
        # logging.info("Deep Ensemble Training Complete!")
        # evaluate_deep_ensemble(test_loader, ENSEMBLE_SIZE)
        # logging.info("Deep Ensemble Evaluation Complete!")


        # # ========================
        # # 📌 UNCERTAINTY ANALYSIS
        # # ========================
        # logging.info("Evaluating Uncertainty for Single Network...")
        # _, _, _, _, _, uncertainties_single, mean_probs_single, all_labels_single = evaluate_mc_dropout(
        #     single_net, test_loader, "SingleNetwork", n_samples=1
        # )
        # mean_preds_single_labels = np.argmax(mean_probs_single, axis=1)  
        # rejection_plot(all_labels_single, mean_preds_single_labels, uncertainties_single, bins=10)

        # ece_single = expected_calibration_error(all_labels_single, mean_probs_single)
        # logging.info(f"ECE for Single Network: {ece_single:.4f}")
        # reliability_diagram(all_labels_single, mean_probs_single)

        
        # # Evaluate Monte Carlo Dropout and obtain uncertainty scores
        # logging.info("Evaluating Monte Carlo Dropout with Rejection Plot...")
        # accuracy_MCD, precision_MCD, recall_MCD, f1_MCD, auc_MCD, uncertainties_MCD, mean_probs_MCD, all_labels_MCD = evaluate_mc_dropout(
        #     mc_dropout_net, test_loader, "MCDropout", n_samples=10
        # )

        # # Convert probabilities to class labels for rejection plot
        # mean_preds_MCD_labels = np.argmax(mean_probs_MCD, axis=1) 

        # # Generate Rejection Plot
        # rejection_plot(all_labels_MCD, mean_preds_MCD_labels, uncertainties_MCD, bins=10)

        # # Compute Calibration Error (ECE) using `mean_probs_MCD`
        # logging.info("Computing Expected Calibration Error (ECE)...")
        # ece_MCD = expected_calibration_error(all_labels_MCD, mean_probs_MCD) 
        # logging.info(f"ECE for MCD: {ece_MCD:.4f}")

        # # Generate Reliability Diagram using `mean_probs_MCD`
        # logging.info("Generating Reliability Diagram for Monte Carlo Dropout...")
        # reliability_diagram(all_labels_MCD, mean_probs_MCD) 

        # # Evaluate Single Network
        # logging.info("Evaluating Single Network...")
        # evaluate_model(single_net, test_loader, "SingleNetwork")
        
