# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from netcal.presentation import ReliabilityDiagram
import json



sns.set(style="whitegrid")

def plot_training_history(history, title='Training and Validation Loss', figsize=(10, 5)):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_history(history, title='Training and Validation Accuracy', figsize=(10, 5)):
    epochs = range(1, len(history['train_acc']) + 1)

    plt.figure(figsize=figsize)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def plot_reliability_diagram(y_true, y_probas, n_bins=10, title='Reliability Diagram'):

    diagram = ReliabilityDiagram(bins=n_bins)
    diagram.plot(y_probas, y_true)
    plt.title(title)
    plt.show()

def plot_entropy_distribution(entropies, title='Entropy Distribution'):
    plt.figure(figsize=(8, 5))
    sns.histplot(entropies, bins=30, kde=True)
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    
def load_results(path_results):
    results = {}
    for method, path in path_results.items():
        if method == 'deep_ensemble':
            ensemble_data = [load_json(p) for p in path]
            results[method] = average_arrays(ensemble_data)
        else:
            results[method] = load_json(path)
    return results

def average_arrays(list_of_dicts):
    """
    Given a list of dictionaries that each contain arrays
    for 'train_loss', 'val_loss', 'train_acc', 'val_acc',
    we average them element-wise for each key.
    """
    if not list_of_dicts:
        return {}
    
    keys = list(list_of_dicts[0].keys())  
    avg_results = {}
    for k in keys:
        arrays = [np.array(d[k]) for d in list_of_dicts if k in d]
        if not arrays:
            avg_results[k] = None
            continue
        stacked = np.stack(arrays, axis=0)
        mean_array = np.mean(stacked, axis=0)
        avg_results[k] = mean_array.tolist()  
    return avg_results


def plot_performance_curves(results):

    methods = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metric_keys = [
        ("train_loss", axes[0,0], "Train Loss"),
        ("val_loss",   axes[0,1], "Val Loss"),
        ("train_acc",  axes[1,0], "Train Accuracy"),
        ("val_acc",    axes[1,1], "Val Accuracy")
    ]
    
    for method in methods:
        data = results[method]
        color = get_color_for_method(method)
        for (metric_name, ax, title) in metric_keys:
            if metric_name not in data or data[metric_name] is None:
                continue
            values = np.array(data[metric_name])
            epochs = np.arange(1, len(values)+1)
            ax.plot(epochs, values, label=method, color=color)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(True)
    
    for (metric_name, ax, title) in metric_keys:
        ax.legend()
    
    plt.tight_layout()
    plt.show()
        

def plot_metrics_performance(results):
    """
    Plots key performance metrics as bar plots for each approach.
    """
    methods = list(results.keys())
    metrics_to_plot = ['accuracy', 'f1_score', 'recall', 'precision', 'ece', 'brier_score']
    num_metrics = len(metrics_to_plot)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
    
    for i, metric in enumerate(metrics_to_plot):
        metric_values = []
        for method in methods:
            value = results[method].get(metric)
            metric_values.append(value if value is not None else 0)
        # Plot as bars
        bars = axes[i].bar(methods, metric_values, color=['blue', 'orange', 'green'])
        axes[i].set_title(metric)
        if metric in ['accuracy', 'f1_score', 'recall', 'precision', 'ece']:
            axes[i].set_ylim(0, 1)
        axes[i].set_ylabel(metric)
        axes[i].set_xticks(range(len(methods)))
        axes[i].set_xticklabels(methods, rotation=45)
        axes[i].grid(True, axis='y')
        
        for bar, value in zip(bars, metric_values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                            f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(results):
    """
    Plots the confusion matrices for each approach using seaborn heatmaps.
    Expects each approach's result to have a key 'confusion_matrix' (as a 2D list/array).
    """
    methods = list(results.keys())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]
    for i, method in enumerate(methods):
        cm = results[method].get("confusion_matrix")
        if cm is None:
            continue
        
        cm = np.array(cm)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f"Confusion Matrix: {method}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
    plt.tight_layout()
    plt.show()
    

def get_color_for_method(method):
    """
    Assign a consistent color for each approach (optional).
    """
    if method == 'single':
        return 'blue'
    elif method == 'mc_dropout':
        return 'orange'
    elif method == 'deep_ensemble':
        return 'green'
    else:
        return 'gray'
    
    

class ReliabilityPlotter:
    def __init__(self, approaches_data):
        self.approaches_data = approaches_data
    
    def plot_all_approaches(self, n_bins=20):
        """
        Pour chaque approche dans approaches_data, trace un unique diagramme de fiabilité (Reliability Diagram).
        n_bins: nombre de bins à utiliser pour le diagramme de fiabilité.
        """
        for approach_name, (y_true, y_probs) in self.approaches_data.items():
            self.plot_reliability(approach_name, y_true, y_probs, n_bins=n_bins)
    
    def plot_reliability(self, approach_name, y_true, y_probs, n_bins=10):
        """
        Trace un unique diagramme de fiabilité (Reliability Diagram) pour l'approche spécifiée,
        en désactivant l'affichage de l'histogramme et de la légende netcal.
        """
        print(f"Plotting reliability diagram for approach: {approach_name}")
        fig = plt.figure(figsize=(6, 5))
        fig.suptitle(f"{approach_name.capitalize()} - Reliability Diagram", fontsize=14, y=0.97)

        rd = ReliabilityDiagram(
            bins=n_bins, 
            show_histogram=True, 
            show_title=True, 
            show_legend=True
        )
        rd.plot(y_probs, y_true)
        
        plt.xlabel("Confiance Moyenne")
        plt.ylabel("Exactitude")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()