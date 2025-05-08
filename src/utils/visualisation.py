# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from netcal.presentation import ReliabilityDiagram
import json
from scipy.stats import ks_2samp
import collections

from matplotlib import pyplot as plt
import numpy as np
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


def load_results_json(method_paths):
    """Charge les métriques finales à partir de fichiers JSON"""
    results = {}
    for method, path in method_paths.items():
        if isinstance(path, list):
            # Deep ensemble
            all_results = []
            for p in path:
                with open(p, 'r') as f:
                    all_results.append(json.load(f))
            # Moyenne des métriques
            metrics = all_results[0].keys()
            avg_result = {
                metric: np.mean([r[metric] for r in all_results if metric in r])
                for metric in metrics
            }
            results[method] = avg_result
        else:
            with open(path, 'r') as f:
                results[method] = json.load(f)
    return results

def plot_performance_curves(path_results):
    """Trace les courbes d'apprentissage (loss & accuracy)"""
    results = {}
    for method, path in path_results.items():
        if isinstance(path, list):
            all_results = []
            for p in path:
                with open(p, 'r') as f:
                    all_results.append(json.load(f))
            avg_results = {}
            for key in all_results[0].keys():
                try:
                    avg_results[key] = np.mean([res[key] for res in all_results if isinstance(res[key], list)], axis=0)
                except:
                    continue
            results[method] = avg_results
        else:
            with open(path, 'r') as f:
                results[method] = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_keys = [
        ("train_loss", axes[0, 0], "Train Loss"),
        ("val_loss",   axes[0, 1], "Val Loss"),
        ("train_acc",  axes[1, 0], "Train Accuracy"),
        ("val_acc",    axes[1, 1], "Val Accuracy")
    ]

    used_axes = set()
    for method, data in results.items():
        color = get_color_for_method(method)
        for (metric_name, ax, title) in metric_keys:
            if metric_name not in data or data[metric_name] is None:
                continue
            values = np.array(data[metric_name])
            if len(values.shape) == 0:
                continue
            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, values, label=method, color=color)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.grid(True)
            used_axes.add(ax)

    for ax in used_axes:
        ax.legend()
    plt.tight_layout()
    plt.show()

def plot_metrics_performance(results):
    """Barplots des métriques globales"""
    methods = list(results.keys())
    metrics_to_plot = ['accuracy', 'f1_score', 'recall', 'precision', 'ece', 'brier_score']
    num_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))

    for i, metric in enumerate(metrics_to_plot):
        metric_values = []
        for method in methods:
            value = results[method].get(metric)
            metric_values.append(value if value is not None else 0)

        bars = axes[i].bar(methods, metric_values, color=[get_color_for_method(m) for m in methods])
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
        
        
def rejection_plot_with_kde_distribution(test_probs, y_test, entropies, num_points=100):
    """
    Trace un diagramme de rejet avec une distribution KDE des entropies.
    test_probs: Probabilités prédites par le modèle.
    y_test: Étiquettes réelles.
    entropies: Entropies calculées pour chaque échantillon.
    num_points: Nombre de points à tracer sur l'axe des x.
    """
    
    entropies = np.array(entropies)
    y_pred = np.argmax(test_probs, axis=1)
    correctness = (y_pred == y_test).astype(int)
    sorted_idx = np.argsort(entropies)
    sorted_entropies = entropies[sorted_idx]
    
    results = {'acc': [], 'rejection_percent': []}
    for percent in np.linspace(0, 1, num_points):
        k = int(percent * len(sorted_entropies))
        selected_idx = sorted_idx[:k]
        acc = np.mean(y_test[selected_idx] == y_pred[selected_idx])
        results['acc'].append(acc)
        results['rejection_percent'].append(percent)
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    ax_left, ax_right = axes
    
    ax_left.plot(results['rejection_percent'], results['acc'], marker='o', label='Accuracy vs Rejection')
    ax_left.set_xlabel('Rejection Percent')
    ax_left.set_ylabel('Accuracy')
    ax_left.set_title("Rejection Plot: Entropy vs Accuracy")
    ax_left.legend()
    ax_left.grid(True, linestyle='--', alpha=0.5)

    correct_ents = entropies[correctness == 1]
    wrong_ents = entropies[correctness == 0]

    mean_correct = np.mean(correct_ents)
    mean_wrong = np.mean(wrong_ents)

    for ents, color, label, mean in [
        (correct_ents, 'blue', 'Correct', mean_correct),
        (wrong_ents, 'red', 'Incorrect', mean_wrong)
    ]:
        sns.kdeplot(ents, shade=True, alpha=0.5, color=color, label=label, ax=ax_right)
        ax_right.axvline(mean, color=color, linestyle='--', label=f'Mean {label}: {mean:.2f}')

    ax_right.set_xlabel("Entropy")
    ax_right.set_ylabel("Density")
    ax_right.set_title("Distribution of Entropies (Correct vs. Incorrect)")
    ax_right.grid(True, linestyle='--', alpha=0.5)
    ax_right.legend()

    # === STATISTIQUES ===
    std_correct = np.std(correct_ents, ddof=1)
    std_wrong = np.std(wrong_ents, ddof=1)
    pooled_std = np.sqrt((std_correct**2 + std_wrong**2) / 2)
    cohen_d = (mean_correct - mean_wrong) / pooled_std

    ks_stat, ks_p = ks_2samp(correct_ents, wrong_ents)

    print(f"Cohen's d = {cohen_d:.3f}")
    print(f"KS test statistic = {ks_stat:.3f}, p-value = {ks_p:.2e}")

    plt.tight_layout()
    plt.show()

    return {
        'entropies': entropies,
        'rejection_curve': results,
        'cohen_d': cohen_d,
        'ks_stat': ks_stat,
        'ks_p': ks_p
    }
    
    
def plot_correct_incorrect_bars(predictor_dict, test_probs, y_test, y_vars, y_entropies, alpha):
    """
    Visualise la distribution des tailles d’ensembles corrects/incorrects avant et après ajustement adaptatif.
    """
    n_methods = len(predictor_dict)
    n_samples = len(y_test)
    y_pred = np.argmax(test_probs, axis=1)
    correctness = (y_test == y_pred).astype(int)
    
    fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 10), sharey='row')
    if n_methods == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, (method_name, predictor) in enumerate(predictor_dict.items()):
        # === 1. Prediction sets
        raw_sets = []
        sorted_classes = []

        for i, (prob, var) in enumerate(zip(test_probs, y_vars)):
            pred_set = predictor.predict(prob.reshape(1, -1))  # shape (1, K)
            raw_sets.append(pred_set[0])
            sorted_classes.append(list(np.argsort(prob)[::-1]))

        # === 2. Analyse AVANT
        set_sizes = [len(s) for s in raw_sets]
        correct_counts, incorrect_counts = collections.defaultdict(int), collections.defaultdict(int)

        for i, s in enumerate(raw_sets):
            is_correct = y_test[i] in s
            key = len(s)
            (correct_counts if is_correct else incorrect_counts)[key] += 1

        _plot_bars(ax=axes[0, col_idx], correct=correct_counts, incorrect=incorrect_counts,
                   title=f"{method_name} (AVANT)", y_test=y_test, pred_sets=raw_sets,
                   alpha=alpha, position='top')

        # === 3. Réajustement adaptatif des sets
        # updated_sets = _adjust_sets_adaptively(raw_sets, sorted_classes, y_entropies)

        # # === 4. Analyse APRÈS
        # updated_sizes = [len(s) for s in updated_sets]
        # correct_counts_post, incorrect_counts_post = collections.defaultdict(int), collections.defaultdict(int)

        # for i, s in enumerate(updated_sets):
        #     is_correct = y_test[i] in s
        #     key = len(s)
        #     (correct_counts_post if is_correct else incorrect_counts_post)[key] += 1

        # _plot_bars(ax=axes[1, col_idx], correct=correct_counts_post, incorrect=incorrect_counts_post,
        #            title=f"{method_name} (APRÈS)", y_test=y_test, pred_sets=updated_sets,
        #            alpha=alpha, position='center')

    plt.tight_layout()
    plt.show()

    
def _plot_bars(ax, correct, incorrect, title, y_test, pred_sets, alpha, position='top'):
    """
    Affiche les barres empilées (correct/incorrect) + stats sur couverture et taille.
    """
    sizes = sorted(set(list(correct.keys()) + list(incorrect.keys())))
    correct_vals = [correct[k] for k in sizes]
    incorrect_vals = [incorrect[k] for k in sizes]

    ax.bar(sizes, correct_vals, color='green', label='Correct')
    ax.bar(sizes, incorrect_vals, bottom=correct_vals, color='red', label='Incorrect')

    for x, c, ic in zip(sizes, correct_vals, incorrect_vals):
        if c > 0:
            ax.text(x, c / 2, str(c), ha='center', va=position, color='white', fontsize=8)
        if ic > 0:
            ax.text(x, c + ic / 2, str(ic), ha='center', va=position, color='white', fontsize=8)

    # Coverage par classe
    n_classes = np.max(y_test) + 1
    coverages_per_class = np.zeros(n_classes)
    counts_per_class = np.zeros(n_classes)
    correct_coverages = []

    for i in range(len(y_test)):
        true_label = y_test[i]
        if true_label in pred_sets[i]:
            coverages_per_class[true_label] += 1
            correct_coverages.append(1)
        else:
            correct_coverages.append(0)
        counts_per_class[true_label] += 1

    coverage_rate = np.mean(correct_coverages)
    class_coverage = coverages_per_class / np.maximum(counts_per_class, 1)
    covgap_avg = np.mean(np.abs(class_coverage - (1 - alpha)))
    vio_classes = np.sum(np.abs(class_coverage - (1 - alpha)) > alpha)

    mean_size = np.mean([len(s) for s in pred_sets])
    std_size = np.std([len(s) for s in pred_sets])
    median_size = np.median([len(s) for s in pred_sets])

    ax.set_title(
        f"{title}\nµ={mean_size:.2f} | σ={std_size:.2f} | M={median_size:.0f}\n"
        f"Coverage={coverage_rate:.2f} | CovGap={covgap_avg:.3f} | Vio={vio_classes}"
    )
    ax.set_xlabel("Set Size")
    ax.set_ylabel("Number of Samples")
    ax.grid(True)
    ax.legend()
    
    
    
def _adjust_sets_adaptively(raw_sets, sorted_classes, entropies):
    """
    Recalibre les ensembles en augmentant dynamiquement leur taille pour les échantillons incertains.
    """
    set_by_size = collections.defaultdict(list)
    for i, (s, sc, ent) in enumerate(zip(raw_sets, sorted_classes, entropies)):
        size = len(s)
        set_by_size[size].append({'index': i, 'entropy': ent, 'set': s, 'sorted_class': sc})

    updated_sets = raw_sets.copy()

    sorted_sizes = sorted(set_by_size.keys())
    for i in range(len(sorted_sizes) - 1):
        k1, k2 = sorted_sizes[i], sorted_sizes[i + 1]
        min_ent_k2 = min(item['entropy'] for item in set_by_size[k2])
        to_transfer = [item for item in set_by_size[k1] if item['entropy'] >= min_ent_k2]

        for item in to_transfer:
            idx = item['index']
            new_set = item['sorted_class'][:k2]
            updated_sets[idx] = new_set
            set_by_size[k2].append(item)

        set_by_size[k1] = [item for item in set_by_size[k1] if item not in to_transfer]

    return updated_sets
