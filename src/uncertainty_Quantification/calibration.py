import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from netcal.binning import HistogramBinning
from netcal.scaling import TemperatureScaling, BetaCalibration
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram

# Import project files
import sys
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # 🔹 Better path handling
sys.path.append(PROJECT_PATH)
from src.uncertainty_Quantification.models import SingleNetwork, MCDropoutNetwork
from utils.utils import enable_dropout






def calibrate_model(probs, labels, method="histogram", bins=10, save_path=None):
    """
    Applies a selected calibration method and returns Expected Calibration Error (ECE).

    Args:
        probs (numpy.array): Model predicted probabilities.
        labels (numpy.array): True labels.
        method (str): Calibration method ("histogram", "temperature", "beta").
        bins (int): Number of bins for binning-based calibration.
        save_path (str or None): If provided, saves the calibrated model.

    Returns:
        float: Expected Calibration Error (ECE).
    """
    # 🔹 Select Calibration Method
    if method == "histogram":
        calibrator = HistogramBinning(bins=bins)
    elif method == "temperature":
        calibrator = TemperatureScaling()
    elif method == "beta":
        calibrator = BetaCalibration()
    else:
        raise ValueError(f"Invalid calibration method: {method}")

    # 🔹 Train & Calibrate in One Step
    calibrated_probs = calibrator.fit_transform(probs, labels)

    # 🔹 Compute ECE
    ece = ECE(bins=bins, equal_intervals=True, norm=True)
    ece_val = ece.measure(calibrated_probs, labels)

    # 🔹 Save the calibration model if required
    if save_path:
        calibrator.save_model(save_path)
        print(f"💾 Calibration model saved: {save_path}")

    # 🔹 Generate Reliability Diagram
    plt.figure(figsize=(6, 6))
    reliability_diagram = ReliabilityDiagram(bins=bins)
    reliability_diagram.plot(calibrated_probs, labels)
    plt.title(f"Reliability Diagram - {method.capitalize()} Calibration")
    plt.savefig(f"{save_path}_reliability.png") if save_path else plt.show()

    return ece_val

# Function to Evaluate Model Calibration
def evaluate_with_calibration(model_type, dataloader, device, methods=["histogram"], bins=10, args=None):
    """
    Evaluate model calibration using different methods.

    Args:
        model_type (str): Model type ("single", "deep_ensemble", "mc_dropout").
        dataloader (torch.utils.data.DataLoader): DataLoader for test dataset.
        device (torch.device): Device to run the model on.
        methods (list): List of calibration methods to use.
        bins (int): Number of bins for binning-based calibration.
        args (Namespace or dict): Additional parameters for deep ensemble and MC Dropout.

    Returns:
        None
    """
    all_preds, all_labels = [], []
    MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, "outputs/models")

    # 🔹 Load Single Model
    if model_type == "single":
        model = SingleNetwork().to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "SingleNetwork.pth"), map_location=device))
        model.eval()

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            with torch.no_grad():
                preds = torch.softmax(model(X_batch), dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    # 🔹 Load Deep Ensemble
    elif model_type == "deep_ensemble":
        models = []
        ensemble_size = args.get("ensemble_size", 5)

        for i in range(ensemble_size):
            model = SingleNetwork().to(device)
            model_path = os.path.join(MODEL_SAVE_PATH, f"DeepEnsemble_{i}.pth")
            if not os.path.exists(model_path):
                print(f"Missing Deep Ensemble model: {model_path}")
                continue
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            with torch.no_grad():
                preds = torch.stack([torch.softmax(model(X_batch), dim=1) for model in models])
                mean_preds = preds.mean(dim=0)
            all_preds.append(mean_preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    elif model_type == "mc_dropout":
        model = MCDropoutNetwork().to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "MCDropout.pth"), map_location=device))
        model.eval()
        enable_dropout(model)  

        num_samples = args.get("num_samples", 50)  

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            with torch.no_grad():
                preds = torch.stack([torch.softmax(model(X_batch), dim=1) for _ in range(num_samples)])
                mean_preds = preds.mean(dim=0)
            all_preds.append(mean_preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    ece_scores = {}
    for method in tqdm(methods, desc="Calibration"):
        ece_scores[method] = calibration_model(all_preds, all_labels, method, bins)
        print(f"🔹 {model_type} - {method.capitalize()} Calibration - ECE: {ece_scores[method]:.4f}")

    os.makedirs(f"{PROJECT_PATH}/outputs/calibration/", exist_ok=True)
    np.savez(f"{PROJECT_PATH}/outputs/calibration/{model_type}_calibration.npz", **ece_scores)
    print(f"✅ Calibration results saved for {model_type} model.")
