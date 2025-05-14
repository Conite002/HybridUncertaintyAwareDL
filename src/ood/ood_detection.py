import torch
import numpy as np
from tqdm import tqdm
import os
import requests
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import gaussian_kde




def get_probs_and_entropy(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    return probs.squeeze().cpu().numpy(), entropy.item()


def download_and_save_image(url, save_path):
    """
    Downloads an image from a URL and saves it to the specified path.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download image from {url}. Code: {response.status_code}")

def create_ood_dataset(image_urls, save_dir):
    """
    Downloads images from a list of URLs and saves them in the specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, url in enumerate(image_urls):
        image_name = f"ood_{i}.jpg"
        save_path = os.path.join(save_dir, image_name)
        download_and_save_image(url, save_path)
        print(f"Downloaded and saved {image_name}")
        
        


def load_image_tensor(image_path):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def get_entropy_single(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
    return entropy.item()

def get_entropy_mcd(model, image_tensor, device, num_passes=10):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = torch.stack([model(image_tensor) for _ in range(num_passes)], dim=0)
        probs = F.softmax(outputs, dim=2)
        mean_probs = probs.mean(dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12), dim=1)
    return entropy.item()

def get_entropy_ensemble(models, image_tensor, device):
    for m in models:
        m.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = torch.stack([m(image_tensor) for m in models], dim=0)
        probs = F.softmax(outputs, dim=2)
        mean_probs = probs.mean(dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12), dim=1)
    return entropy.item()

def compute_entropy_scores(model_or_models, image_dir, device, approach="single", num_passes=10):
    scores = []
    for fname in os.listdir(image_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            image_path = os.path.join(image_dir, fname)
            tensor = load_image_tensor(image_path)
            if approach == "single":
                entropy = get_entropy_single(model_or_models, tensor, device)
            elif approach == "mc_dropout":
                entropy = get_entropy_mcd(model_or_models, tensor, device, num_passes)
            elif approach == "deep_ensemble":
                entropy = get_entropy_ensemble(model_or_models, tensor, device)
            else:
                raise ValueError("Unsupported approach")
            scores.append(entropy)
    return np.array(scores)



def evaluate_ood_entropy_all(models_dict, id_dir, ood_dir, device, num_passes=5):
    """
    Plot entropy distributions for multiple models in one row with 3 columns.
    
    models_dict: dict like {
        "single": model,
        "mc_dropout": model,
        "deep_ensemble": [model1, model2, ...]
    }
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (approach, model_or_models) in enumerate(models_dict.items()):
        id_scores = compute_entropy_scores(model_or_models, id_dir, device, approach, num_passes)
        ood_scores = compute_entropy_scores(model_or_models, ood_dir, device, approach, num_passes)

        y_true = np.array([0]*len(id_scores) + [1]*len(ood_scores))
        y_scores = np.concatenate([id_scores, ood_scores])

        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)

        kde_id = gaussian_kde(id_scores)
        kde_ood = gaussian_kde(ood_scores)
        x_range = np.linspace(0, max(y_scores) + 0.5, 500)

        axs[i].plot(x_range, kde_id(x_range), label='ID (In-Distribution)', linewidth=2)
        axs[i].plot(x_range, kde_ood(x_range), label='OOD (Out-of-Distribution)', linewidth=2)
        axs[i].fill_between(x_range, kde_id(x_range), alpha=0.3)
        axs[i].fill_between(x_range, kde_ood(x_range), alpha=0.3)
        axs[i].set_title(f"{approach} (AUROC={auroc:.2f}, AUPRC={auprc:.2f})")
        axs[i].set_xlabel("Predictive Entropy")
        axs[i].set_ylabel("Density")
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    plt.show()
