import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import os, sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)


from src.uncertainty_Quantification.losses import relu_evidence
from src.uncertainty_Quantification.helpers import rotate_img, one_hot_embedding, get_device
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.uncertainty_Quantification.models import SingleNetwork
import logging
from utils.utils import enable_dropout



DEVICE = get_device()
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs/models")
HISTORY_SAVE_PATH = os.path.join(PROJECT_ROOT, "outputs/results")


def test_single_image(model, img_path, uncertainty=False, device=None):
    img = Image.open(img_path).convert("L")
    if not device:
        device = get_device()
    num_classes = 10
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    img_tensor = trans(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    img_variable = img_variable.to(device)

    if uncertainty:
        output = model(img_variable)
        evidence = relu_evidence(output)
        alpha = evidence + 1
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        _, preds = torch.max(output, 1)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob)
        print("Uncertainty:", uncertainty)

    else:

        output = model(img_variable)
        _, preds = torch.max(output, 1)
        prob = F.softmax(output, dim=1)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob)

    labels = np.arange(10)
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})

    plt.title("Classified as: {}, Uncertainty: {}".format(preds[0], uncertainty.item()))

    axs[0].set_title("One")
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")

    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    axs[1].set_xlim([0, 9])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Classification Probability")

    fig.tight_layout()

    plt.savefig("/Outputs/results{}".format(os.path.basename(img_path)))


def rotating_image_classification(
    model, img, filename, uncertainty=False, threshold=0.5, device=None
):
    if not device:
        device = get_device()
    num_classes = 10
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    classifications = []

    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)

        nimg = np.clip(a=nimg, a_min=0, a_max=1)

        rimgs[:, i * 28 : (i + 1) * 28] = nimg
        trans = transforms.ToTensor()
        img_tensor = trans(nimg)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        img_variable = img_variable.to(device)

        if uncertainty:
            output = model(img_variable)
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            _, preds = torch.max(output, 1)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())
            lu.append(uncertainty.mean())

        else:

            output = model(img_variable)
            _, preds = torch.max(output, 1)
            prob = F.softmax(output, dim=1)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())

        scores += prob.detach().cpu().numpy() >= threshold
        ldeg.append(deg)
        lp.append(prob.tolist())

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ["black", "blue", "red", "brown", "purple", "cyan"]
    marker = ["s", "^", "o"] * 2
    labels = labels.tolist()
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [4, 1, 12]})

    for i in range(len(labels)):
        axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    if uncertainty:
        labels += ["uncertainty"]
        axs[2].plot(ldeg, lu, marker="<", c="red")

    print(classifications)

    axs[0].set_title('Rotated "1" Digit Classifications')
    axs[0].imshow(1 - rimgs, cmap="gray")
    axs[0].axis("off")
    plt.pause(0.001)

    empty_lst = []
    empty_lst.append(classifications)
    axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
    axs[1].axis("off")

    axs[2].legend(labels)
    axs[2].set_xlim([0, Mdeg])
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel("Rotation Degree")
    axs[2].set_ylabel("Classification Probability")

    plt.savefig(filename)
    

def plot_confusion_matrix(y_true, y_pred, num_classes):
    """
    Plots the confusion matrix.
    
    Args:
    - y_true (np.array): True labels.
    - y_pred (np.array): Predicted labels.
    - num_classes (int): Number of classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()




# 📌 Evaluate Model
def evaluate_model(model, test_loader, model_name):
    model.to(DEVICE)
    # load model
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth")))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = model(X_batch)
            
            probs = torch.softmax(outputs, dim=1)            
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    auc = roc_auc_score(np.eye(5)[all_labels], np.eye(5)[all_preds], multi_class="ovr")

    logging.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    result = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
    }
    return result


def evaluate_deep_ensemble(test_loader, ensemble_size=5):
    logging.info("Evaluating Deep Ensemble...")
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
            mean_preds = ensemble_preds.mean(dim=0) 
            final_preds = torch.argmax(mean_preds, dim=1) 

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Compute ensemble evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    auc = roc_auc_score(np.eye(5)[all_labels], np.eye(5)[all_preds], multi_class="ovr")

    logging.info(f"Deep Ensemble - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    result = {
        "model": "Deep Ensemble",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
    }
    return result
    
    
    
def monte_carlo_dropout(model, dataloader, n_samples=10, device=DEVICE, model_name="MCDropout"):
    """
    Perform Monte Carlo Dropout Inference.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for dataset.
        n_samples (int): Number of forward passes.
        device (str): "cuda" or "cpu".

    Returns:
        - mean_preds (np.array): Mean class predictions.
        - mean_probs (np.array): Mean predicted probabilities.
        - uncertainty (np.array): Variance of predictions (uncertainty).
        - all_labels (np.array): True labels.
    """
    model.to(device)
    model.eval()
    enable_dropout(model)
    

    all_probs = []
    all_labels = []
    all_uncertainties = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = torch.stack([torch.softmax(model(X_batch), dim=1) for _ in range(n_samples)])

            mean_probs = preds.mean(dim=0)  
            uncertainty = preds.var(dim=0).sum(dim=1)  
            mean_preds = torch.argmax(mean_probs, dim=1) 

            # ✅ Move tensors to CPU before converting to NumPy
            all_probs.append(mean_probs.cpu().numpy()) 
            all_uncertainties.append(uncertainty.cpu().numpy()) 
            all_labels.append(y_batch.cpu().numpy()) 

    # save model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth"))
    return np.concatenate(all_probs), np.concatenate(all_probs), np.concatenate(all_uncertainties), np.concatenate(all_labels)


def evaluate_mc_dropout(model, test_loader, model_name, n_samples=10):
    """
    Evaluate model using Monte Carlo Dropout.

    Args:
        model (torch.nn.Module): Trained Monte Carlo Dropout model.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        model_name (str): Name of the model.
        n_samples (int): Number of forward passes.

    Returns:
        - accuracy (float)
        - precision (float)
        - recall (float)
        - f1 (float)
        - auc (float)
        - uncertainties (np.array)
        - mean_probs (np.array)
        - all_labels (np.array)
    """
    model.to(DEVICE)
    model.eval()
    
    logging.info(f"Evaluating {model_name} with Monte Carlo Dropout ({n_samples} samples)...")
    # load model weights
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, f"{model_name}.pth")))
    # ✅ Apply Monte Carlo Dropout with Uncertainty Estimation
    mean_preds, mean_probs, uncertainties, all_labels = monte_carlo_dropout(model, test_loader, n_samples, DEVICE, model_name)

    # ✅ Move all outputs to CPU before converting to NumPy
    mean_preds = mean_preds if isinstance(mean_preds, np.ndarray) else mean_preds.cpu().numpy()
    mean_probs = mean_probs if isinstance(mean_probs, np.ndarray) else mean_probs.cpu().numpy()
    uncertainties = uncertainties if isinstance(uncertainties, np.ndarray) else uncertainties.cpu().numpy()
    all_labels = all_labels if isinstance(all_labels, np.ndarray) else all_labels.cpu().numpy()

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, np.argmax(mean_probs, axis=1))  
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, np.argmax(mean_probs, axis=1), average="macro")
    auc = roc_auc_score(np.eye(5)[all_labels], mean_probs, multi_class="ovr")  

    logging.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "uncertainties": uncertainties,
    }
    return results
