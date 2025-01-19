import numpy as np
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from pandas import DataFrame


def plot_spectrogram(
    data_frame: DataFrame,
    column: str = "vibration_1_magnitude",
    aspect: str = None,
    cmap="viridis",
    figsize=(12, 6),
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
        values=column, index="rpm", columns="fft_frequency", fill_value=0
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
