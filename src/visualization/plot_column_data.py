import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from pandas import DataFrame


def plot_column_data(
    dataframe: DataFrame, columns: list, plot_type: str = "scatter", dpi=100
):
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
    if not isinstance(dataframe, DataFrame):
        raise TypeError("data_frame must be a pandas DataFrame.")

    if not all(col in dataframe.columns for col in columns):
        raise ValueError("Some columns are not present in the DataFrame.")

    n_plots = len(columns)
    n_cols = 2
    n_rows = ceil(n_plots / n_cols)

    figure_size = (1920 / dpi, 1080 / dpi)
    scaling_factor = dpi / 100

    figure, axes = plt.subplots(
        n_rows, n_cols, figsize=figure_size, dpi=dpi, constrained_layout=True
    )

    if axes.ndim != 1:
        axes = axes.flatten()

    for i, column in enumerate(columns):
        axis = axes[i]
        match plot_type:
            case "plot":
                axis.plot(dataframe[column], linewidth=scaling_factor)
            case "scatter":
                axis.scatter(
                    x=dataframe[column].index,
                    y=dataframe[column],
                    s=scaling_factor,
                    edgecolors="face",
                )
            case _:
                raise ValueError(f"Invalid plot type '{plot_type}'!")

        axis.set_title(f"'{column}' Data")

        axis.set_ylabel(column, fontsize=12 * scaling_factor)
        axis.set_xlabel(dataframe.attrs["index_type"])
        axis.grid(True)

    for i in range(len(columns), len(axes)):
        figure.delaxes(axes[i])

    # set figure super title
    figure_name = dataframe.attrs["path"].stem
    unbalance = dataframe.attrs["unbalance"].title()
    figure.suptitle(f"Dataset {figure_name}:    {unbalance} Unbalance")

    plt.show()
