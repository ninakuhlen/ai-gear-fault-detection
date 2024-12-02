from matplotlib.pyplot import subplots, show
from pandas import DataFrame
from math import ceil


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

    fig, axes = subplots(
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

    show()
