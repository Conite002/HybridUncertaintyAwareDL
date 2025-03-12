
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, test_data, model_name):
    """Evaluates a trained model and prints a classification report & confusion matrix."""
    y_true = test_data.classes
    y_pred = np.argmax(model.predict(test_data), axis=1)

    print(f"\n🔍 Classification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    plt.show()
