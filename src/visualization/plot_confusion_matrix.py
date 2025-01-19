# from matplotlib.pyplot import subplots, show
from fnmatch import fnmatch
import matplotlib.pyplot as plt
from pandas import DataFrame
from math import ceil
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Plots a confusion matrix using matplotlib.

    Args:
        true_labels (array-like): The true labels (ground truth) from the dataset.
        predicted_labels (array-like): The predicted labels from the model.
        class_names (list): List of class names for the matrix.
    """

    cm = confusion_matrix(
        true_labels, predicted_labels
    )  # labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 8))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()
