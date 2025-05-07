import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# import compute_ece



def multiclass_brier(y_true, y_proba):
    n_samples = len(y_true)
    n_classes = y_proba.shape[1]
    y_true_onehot = np.zeros_like(y_proba)
    y_true_onehot[np.arange(n_samples), y_true] = 1
    return np.mean(np.sum((y_proba - y_true_onehot)**2, axis=1))

def compute_ece(y_true, y_proba, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba.max(axis=1), bin_boundaries) - 1
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_probs = y_proba[bin_indices == i]
            bin_true = y_true[bin_indices == i]
            bin_accuracy = np.mean(np.argmax(bin_probs, axis=1) == bin_true)
            bin_confidence = np.mean(bin_probs.max(axis=1))
            ece += (bin_counts[i] / len(y_true)) * abs(bin_accuracy - bin_confidence)
    
    return ece
    
def entropy(y_proba):
    return -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
    
def compute_metrics(y_true, y_pred, y_proba, y_variance=None):
    res = {}
    
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    res['precision'] = precision_score(y_true, y_pred, average='weighted')
    res['recall'] = recall_score(y_true, y_pred, average='weighted')
    res['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
    
    res['brier_score'] = multiclass_brier(y_true, y_proba)
    res['entropy'] = entropy(y_proba)
    res['uncertainty'] = np.std(y_proba, axis=1)
    res['ece'] = compute_ece(y_true, y_proba)
    res['variance'] = y_variance
    return res

