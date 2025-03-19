import torch
import torch.nn as nn
import torch.optim as optim         
import os, sys
import numpy as np
from matplotlib import pyplot as plt

import json, logging
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

from helpers import get_device, rotate_img, one_hot_embedding
from train import train_model, train_deep_ensemble, get_dataloader
from test import evaluate_model, evaluate_deep_ensemble, evaluate_mc_dropout
from models import SingleNetwork, MCDropoutNetwork
from utils.utils import enable_dropout, set_seed
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from calibration import expected_calibration_error, reliability_diagram

from uncertainty_analysis import rejection_plot
from utils.utils import get_device, save_history
DEVICE = get_device()


set_seed(42)
# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE =10
ENSEMBLE_SIZE = 5

# Load DataLoaders
train_loader = get_dataloader("train", BATCH_SIZE)
val_loader = get_dataloader("val", BATCH_SIZE)

# Initialize model
input_dim = train_loader.dataset.tensors[0].shape[1]
num_classes = len(set(train_loader.dataset.tensors[1].numpy()))

######################
# 📌 Single Network  #
######################
# logging.info("Training Single Network...")
single_net = SingleNetwork(input_dim, num_classes).to(DEVICE)
optimizer = torch.optim.Adam(single_net.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()
train_model(single_net, train_loader, val_loader, criterion, optimizer, EPOCHS, "SingleNetwork", PATIENCE)

######################
#  📌 Deep Ensemble  #
######################
logging.info(f"Training Deep Ensemble with {ENSEMBLE_SIZE} models...")
ensemble_models = train_deep_ensemble(
    train_loader, val_loader, ensemble_size=ENSEMBLE_SIZE, 
    learning_rate=LEARNING_RATE, epochs=EPOCHS
)

##########################
# 📌 Monte Carlo Dropout #
##########################
logging.info("Training Monte Carlo Dropout Network...")
mc_dropout_net = MCDropoutNetwork(input_dim, num_classes, dropout_rate=0.5).to(DEVICE)
optimizer = torch.optim.Adam(mc_dropout_net.parameters(), lr=LEARNING_RATE)
train_model(mc_dropout_net, train_loader, val_loader, criterion, optimizer, EPOCHS, "MCDropout", PATIENCE)



# # Evaluate Models
# test_loader = get_dataloader("test", BATCH_SIZE)
# evaluate_model(single_net, test_loader, "SingleNetwork")
# evaluate_deep_ensemble(test_loader, ENSEMBLE_SIZE)
# evaluate_mc_dropout(mc_dropout_net, test_loader, "MCDropout", n_samples=10)



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

    