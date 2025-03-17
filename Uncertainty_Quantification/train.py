import os, sys
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import json
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_ROOT)
from Uncertainty_Quantification.models import SingleNetwork, MCDropoutNetwork, DeepEnsemble
from Uncertainty_Quantification.helpers import get_device
# 📌 Configuration



DEVICE = get_device()
FEATURES_PATH = os.path.join(PROJECT_ROOT, "Feature_Extraction")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "Outputs/models")
HISTORY_SAVE_PATH = os.path.join(PROJECT_ROOT, "Outputs/results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(HISTORY_SAVE_PATH, exist_ok=True)

# 📌 Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📌 Load extracted features
def load_features(split):
    features = np.load(f"{FEATURES_PATH}/ResNet50_{split}.npy")
    labels = np.load(f"{FEATURES_PATH}/labels_{split}.npy")
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# 📌 Create DataLoader
def get_dataloader(split, batch_size):
    X, y = load_features(split)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train" or split == "val"))


# 📌 Train Model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, learning_rate, model_name, patience=5):
    model.to(DEVICE)
    best_val_loss = float("inf") 
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0 
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * y_batch.size(0)
                val_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        logging.info(f"{model_name} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model based on validation accuracy
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            patience_counter = 0
            logging.info(f"✅ New best model found for {model_name}, saving model...")
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     best_model_state = model.state_dict()
        #     best_epoch = epoch + 1
        #     patience_counter = 0  
            # logging.info(f"✅ New best model found for {model_name}, saving model...")
        else:
            patience_counter += 1
            logging.info(f"⏳ No improvement in validation loss for {patience_counter}/{patience} epochs.")

        # Stop training if patience is exceeded
        if patience_counter >= patience:
            logging.info(f"⏹️ Early stopping activated for {model_name}. Stopping training.")
            break
    if best_model_state:
        torch.save(best_model_state, os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth"))
        logging.info(f"💾 Best model saved for {model_name}  at epochs {best_epoch} with val_loss={best_val_loss:.4f}")

    with open(os.path.join(HISTORY_SAVE_PATH, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f)


def train_deep_ensemble(train_loader, val_loader, ensemble_size=5, learning_rate=0.0005, epochs=30):
    ensemble_models = []

    for i in range(ensemble_size):
        logging.info(f"Training Deep Ensemble Model {i+1}/{ensemble_size}")
        model = SingleNetwork().to(DEVICE)  
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_model(model, train_loader, val_loader, criterion, optimizer, epochs, learning_rate=learning_rate, model_name=f"DeepEnsemble_{i}")
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"deep_ensemble_{i}.pth"))

        ensemble_models.append(model)

    logging.info("Deep Ensemble Training Complete!")
    return ensemble_models
