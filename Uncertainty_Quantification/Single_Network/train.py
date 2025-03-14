import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence
from tqdm import tqdm
import torch.nn.functional as F
import copy
import json
import os


def enable_dropout(m):
    """
    Enables Dropout layers at inference time for Monte Carlo Dropout.
    This is necessary because PyTorch disables Dropout during evaluation mode.
    """
    if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d) or isinstance(m, torch.nn.Dropout3d):
        m.train()



def train_model(model, dataloaders, num_classes, criterion, optimizer, scheduler=None, num_epochs=25, device=None, uncertainty_method="single"):
    """
    Trains a model using extracted features with optional uncertainty-aware classification.

    Returns:
    - model: The trained PyTorch model with the best validation accuracy.
    - metrics: A dictionary containing loss and accuracy per epoch for both training and validation.
    """
    since = time.time()
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # --- Training Phase ---
        model.train()
        if uncertainty_method == "mc_dropout":
            model.apply(enable_dropout)  

        running_loss = 0.0
        running_corrects = 0
        total = 0

        data_loader = dataloaders["train"]

        with tqdm(total=len(data_loader), desc="Training Progress") as pbar:
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    if uncertainty_method in ["deep_ensemble", "mc_dropout"]:
                        labels = labels.to(device)
                        y = one_hot_embedding(labels, num_classes).to(device)

                        outputs = model(inputs, activation="softplus")
                        loss = criterion(outputs, y.float(), epoch_num=epoch, num_classes=num_classes, annealing_step=5, device=device)

                        preds = torch.argmax(outputs, dim=1)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        total_evidence = torch.sum(evidence, dim=1, keepdim=True) + 1e-8
                        mean_evidence = torch.mean(total_evidence)

                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += labels.size(0)

                pbar.update(1)

        epoch_train_loss = running_loss / len(data_loader.dataset)
        epoch_train_acc = running_corrects / total  

        metrics["train_loss"].append(epoch_train_loss)
        metrics["train_accuracy"].append(epoch_train_acc)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Accuracy: {epoch_train_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders["val"]):
                inputs, labels = inputs.to(device), labels.to(device)

                if uncertainty_method in ["deep_ensemble", "mc_dropout"]:
                    y = one_hot_embedding(labels, num_classes).to(device)
                    outputs = model(inputs, activation="softplus")
                    loss = criterion(outputs, y.float(), epoch_num=-1, num_classes=num_classes, annealing_step=5, device=device)  # 🔥 Fix: Use epoch_num=-1
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item() * inputs.shape[0] 
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects / val_total  
        

        metrics["val_loss"].append(epoch_val_loss)
        metrics["val_accuracy"].append(epoch_val_acc)

        print(f"Validation Loss: {epoch_val_loss:.4f} | Validation Accuracy: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()  

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s | Best Validation Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    return model, metrics



################################################################################################################


import torch
import torch.optim as optim
import torch.nn as nn
import copy
import time
import json
import os
from tqdm import tqdm
from helpers import get_device, one_hot_embedding
from losses import relu_evidence, edl_mse_loss, edl_log_loss, edl_digamma_loss


def enable_dropout(m):
    """
    Active Dropout pendant l'inférence pour MC Dropout.
    """
    if isinstance(m, torch.nn.Dropout):
        m.train()


def train_single_model(model, dataloaders, num_classes, criterion, optimizer, scheduler=None, num_epochs=25, device=None):
    """
    Entraîne un seul modèle avec une seule loss.
    """
    since = time.time()
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in tqdm(dataloaders["train"], desc="Training Progress"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_train_loss = running_loss / len(dataloaders["train"].dataset)
        epoch_train_acc = running_corrects / total  

        metrics["train_loss"].append(epoch_train_loss)
        metrics["train_accuracy"].append(epoch_train_acc)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects / val_total  

        metrics["val_loss"].append(epoch_val_loss)
        metrics["val_accuracy"].append(epoch_val_acc)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler:
            scheduler.step()  

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s | Best Validation Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    return model, metrics



def train_deep_ensemble(num_models, model_class, dataloaders, num_classes, criterion, optimizer_fn, scheduler_fn, num_epochs=25, device=None):
    """
    Entraîne un Deep Ensemble (plusieurs modèles indépendants).
    """
    ensemble = []
    for i in range(num_models):
        print(f"\n🔹 Training Deep Ensemble Model {i+1}/{num_models}...")

        model = model_class().to(device)
        optimizer = optimizer_fn(model.parameters())
        scheduler = scheduler_fn(optimizer)

        trained_model, metrics = train_single_model(model, dataloaders, num_classes, criterion, optimizer, scheduler, num_epochs, device)
        ensemble.append(trained_model)

    return ensemble
from test import evaluate_model


def train_mc_dropout(model, dataloaders, num_classes, criterion, optimizer, scheduler=None, num_epochs=25, device=None):
    """
    Entraîne un modèle avec Monte Carlo Dropout (Dropout activé à l'inférence).
    """
    model.apply(enable_dropout)  # Active Dropout en mode eval aussi
    return train_single_model(model, dataloaders, num_classes, criterion, optimizer, scheduler, num_epochs, device)



def benchmark_models(models, dataloaders, num_classes, device):
    """
    Évalue la performance de chaque modèle sur le test set.
    """
    results = []
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        model.eval()
        test_loss, test_acc = evaluate_model(model, dataloaders["test"], num_classes, device)
        results.append({"Method": model_name, "Test Accuracy": test_acc, "Test Loss": test_loss})

    return results

