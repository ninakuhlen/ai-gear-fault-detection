import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(true_labels, predicted_labels, class_names) -> dict:
    """
    Plots a confusion matrix using matplotlib.

    Args:
        true_labels (array-like): The true labels (ground truth) from the dataset.
        predicted_labels (array-like): The predicted labels from the model.
        class_names (list): List of class names for the matrix.
    """

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    confuxion_matrix_plot = plt.figure(figsize=(14, 14))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    file_name = "confusion_matrix"
    return {"figure": confuxion_matrix_plot, "file_name": file_name}
