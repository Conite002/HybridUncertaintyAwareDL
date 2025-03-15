import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import json
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import logging

# 📌 Import Models
from models import SingleNetwork, MCDropoutNetwork, DeepEnsemble

# 📌 Configuration
FEATURE_PATH = "../Feature_Extraction/ResNet50"
MODEL_SAVE_PATH = "../models"
HISTORY_SAVE_PATH = "../training_history"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 30
ENSEMBLE_SIZE = 5
LEARNING_RATE = 0.0005

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(HISTORY_SAVE_PATH, exist_ok=True)

# 📌 Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 📌 Load extracted features
def load_features(split):
    features = np.load(f"{FEATURE_PATH}/ResNet50_{split}.npy")
    labels = np.load(f"{FEATURE_PATH}/labels_{split}.npy")
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# 📌 Create DataLoader
def get_dataloader(split, batch_size):
    X, y = load_features(split)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train" or split == "val"))

# 📌 Train Model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_name):
    model.to(DEVICE)
    best_val_acc = 0
    best_model_state = None

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        # Training Loop
        for X_batch, y_batch in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation Loop
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        logging.info(f"{model_name} Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

    # Save best model
    if best_model_state:
        torch.save(best_model_state, os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth"))

    # Save training history
    with open(os.path.join(HISTORY_SAVE_PATH, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f)

# 📌 Evaluate Model
def evaluate_model(model, test_loader, model_name):
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    auc = roc_auc_score(np.eye(5)[all_labels], np.eye(5)[all_preds], multi_class="ovr")

    logging.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    return accuracy, precision, recall, f1, auc


# 📌 Train Deep Ensemble
def train_deep_ensemble(train_loader, val_loader, ensemble_size=5):
    ensemble_models = []

    for i in range(ensemble_size):
        logging.info(f"Training Deep Ensemble Model {i+1}/{ensemble_size}")
        model = SingleNetwork().to(DEVICE)  
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, f"DeepEnsemble_{i}")
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"deep_ensemble_{i}.pth"))

        ensemble_models.append(model)

    logging.info("Deep Ensemble Training Complete!")
    return ensemble_models

def evaluate_deep_ensemble(test_loader, ensemble_size=5):
    models = []
    for i in range(ensemble_size):
        model = SingleNetwork().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f"DeepEnsemble_{i}.pth")))
        model.eval()
        models.append(model)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            # Collect predictions from all ensemble models
            ensemble_preds = torch.stack([torch.softmax(model(X_batch), dim=1) for model in models])
            mean_preds = ensemble_preds.mean(dim=0)  # Take mean prediction
            final_preds = torch.argmax(mean_preds, dim=1)  # Take final predicted class

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Compute ensemble evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    auc = roc_auc_score(np.eye(5)[all_labels], np.eye(5)[all_preds], multi_class="ovr")

    logging.info(f"Deep Ensemble - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    return accuracy, precision, recall, f1, auc



# 📌 Main Training Pipeline
if __name__ == "__main__":
    logging.info("Loading Data...")
    train_loader = get_dataloader("train", BATCH_SIZE)
    val_loader = get_dataloader("val", BATCH_SIZE)
    test_loader = get_dataloader("test", BATCH_SIZE)

    # 📌 Train Single Network
    logging.info("Training Single Network...")
    single_net = SingleNetwork().to(DEVICE)
    optimizer = optim.Adam(single_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_model(single_net, train_loader, val_loader, criterion, optimizer, EPOCHS, "SingleNetwork")
    evaluate_model(single_net, test_loader, "SingleNetwork")

    torch.save(single_net.state_dict(), os.path.join(MODEL_SAVE_PATH, "single_network.pth"))
    logging.info("Single Network Training Complete!")

    # 📌 Train Monte Carlo Dropout Network
    logging.info("Training Monte Carlo Dropout Network...")
    mc_dropout_net = MCDropoutNetwork().to(DEVICE)
    optimizer = optim.Adam(mc_dropout_net.parameters(), lr=LEARNING_RATE)

    train_model(mc_dropout_net, train_loader, val_loader, criterion, optimizer, EPOCHS, "MCDropout")
    evaluate_model(mc_dropout_net, test_loader, "MCDropout")

    torch.save(mc_dropout_net.state_dict(), os.path.join(MODEL_SAVE_PATH, "mc_dropout.pth"))
    logging.info("Monte Carlo Dropout Training Complete!")

    logging.info("Training Deep Ensemble...")
    train_deep_ensemble(train_loader, val_loader, ENSEMBLE_SIZE)
    evaluate_deep_ensemble(test_loader, ENSEMBLE_SIZE)
    logging.info("Deep Ensemble Evaluation Complete!")