# from matplotlib.pyplot import subplots, show
import matplotlib.pyplot as plt
from pandas import DataFrame
from math import ceil
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


def plot_columns_as_subplots(data_frame: DataFrame, columns: list, dpi=100):
    """
    Plots selected columns of a pandas DataFrame as subplots in a 2-column layout, optimized for high dpi values without visual distortion.

    Args:
        data_frame (DataFrame): The DataFrame containing the data to plot.
        columns (list): List of column names to be plotted.
        dpi (int, optional): Dots per inch for the figure resolution. Defaults to 100.

    Raises:
        TypeError: data_frame must be a pandas DataFrame.
        ValueError: Some columns are not present in the DataFrame.
    """
    if not isinstance(data_frame, DataFrame):
        raise TypeError("data_frame must be a pandas DataFrame.")

    if not all(col in data_frame.columns for col in columns):
        raise ValueError("Some columns are not present in the DataFrame.")

    n_plots = len(columns)
    n_cols = 2
    n_rows = ceil(n_plots / n_cols)

    figure_size = (1920 / dpi, 1080 / dpi)
    scaling_factor = dpi / 100

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figure_size, dpi=dpi, constrained_layout=True
    )

    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        ax.plot(data_frame[col], label=col, linewidth=1.5 * scaling_factor)
        ax.set_title(col, fontsize=12 * scaling_factor)
        ax.grid(True)

    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])

    # TODO Gemeinsame Achsenbeschriftung
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Index", fontsize=12 * scaling_factor)

    plt.show()


def plot_fft_spectrogram(
    data_frame: DataFrame, aspect: str = None, cmap="viridis", figsize=(12, 6)
):
    """
    Plots a pixel graphic (heatmap) of FFT data with frequency on the X-axis, RPM on the Y-axis,
    and magnitude as color intensity.

    Args:
        data_frame (DataFrame): Input DataFrame with FFT data.
        aspect (str, optional): Aspect ratio for the heatmap. Defaults to 'auto'.
        cmap (str, optional): Colormap for the plot. Defaults to "viridis".
        figsize (tuple, optional): Figure size (width, height). Defaults to (12, 6).

    Raises:
        ValueError: If required columns are not in the DataFrame.
    """

    if aspect is None:
        aspect = "auto"

    # Pivot-Tabelle erstellen
    pivot_data = data_frame.pivot_table(
        values="fft_magnitude", index="rpm", columns="fft_frequency", fill_value=0
    )

    # Konvertieren in Matrix
    rpm_values = pivot_data.index
    freq_values = pivot_data.columns
    amplitude_matrix = pivot_data.values

    # Logarithmische Skalierung der Amplituden (optional) für bessere Sichtbarkeit
    amplitude_matrix = np.log1p(
        amplitude_matrix
    )  # Logarithmische Transformation: log(1 + x)

    # Grafik erstellen
    plt.figure(figsize=figsize)  # Breitere Darstellung
    plt.imshow(
        amplitude_matrix,
        aspect=aspect,  # Automatische Anpassung der Darstellung
        cmap=cmap,  # Farbskala # 'inferno'
        origin="lower",  # Startet bei der unteren linken Ecke
        extent=[
            freq_values.min(),
            freq_values.max(),
            rpm_values.min(),
            rpm_values.max(),
        ],
        vmin=np.percentile(amplitude_matrix, 5),
        vmax=np.percentile(amplitude_matrix, 95),
    )

    # Achsentitel und Farblegende hinzufügen
    plt.colorbar(label="Log-scaled FFT Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Rotation Speed [rpm]")
    plt.title("FFT of Vibration Data - Transformed Test")

    plt.show()


def plot_training_history(history, metrics=["loss", "accuracy", "precision", "recall"]):
    """
    Visualizes the metrics of the training.

    Args:
        history: The History-Object from Training (model.fit).
        metrics (list): List of the metrics that should be plotted. Default: loss, accuracy, precision, recall.
    """
    plt.figure(figsize=(12, 8))

    for metric in metrics:
        if metric in history.history:
            plt.plot(history.history[metric], label=f"Train {metric}")
            if f"val_{metric}" in history.history:
                plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training and Validation")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Plots a confusion matrix using matplotlib.

    Args:
        true_labels (array-like): The true labels (ground truth) from the dataset.
        predicted_labels (array-like): The predicted labels from the model.
        class_names (list): List of class names for the matrix.
    """

    # encode string labels to integers
    label_encoder = LabelEncoder()
    true_labels_encoded = label_encoder.fit_transform(true_labels)
    predicted_labels_encoded = label_encoder.transform(predicted_labels)

    cm = confusion_matrix(
        true_labels_encoded, predicted_labels_encoded, labels=range(len(class_names))
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 8))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()
