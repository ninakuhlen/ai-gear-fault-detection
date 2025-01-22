import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from pandas import DataFrame


def plot_column_data(
    dataframe: DataFrame,
    columns: list,
    plot_type: str = "scatter",
    dpi=100,
    separated: bool = False,
    return_only: bool = False,
) -> list:
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

    figure_dicts = []

    if separated:
        for column in columns:

            data_plot = plt.figure(figsize=(9, 6))
            match plot_type:
                case "plot":
                    plt.plot(dataframe[column])
                case "scatter":
                    plt.scatter(
                        x=dataframe[column].index,
                        y=dataframe[column],
                        edgecolors="face",
                    )
                case _:
                    raise ValueError(f"Invalid plot type '{plot_type}'!")

            plt.title(f"'{column}' Data", fontsize="xx-large")
            plt.ylabel(column, fontsize="x-large")
            plt.xlabel(dataframe.attrs["index_type"], fontsize="x-large")
            plt.grid(True)

            if not return_only:
                plt.show()
            else:
                plt.close(data_plot)

            file_name = dataframe.attrs["path"].stem + f"_{column}"
            figure_dict = {"figure": data_plot, "file_name": file_name}
            figure_dicts.append(figure_dict)

    else:
        n_plots = len(columns)
        n_cols = 2
        n_rows = ceil(n_plots / n_cols)

        figure_size = (1920 / dpi, 1080 / dpi)
        scaling_factor = dpi / 100

        data_plot, axes = plt.subplots(
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
            axis.set_xlabel(dataframe.attrs["index_type"], fontsize=12 * scaling_factor)
            axis.grid(True)

        for i in range(len(columns), len(axes)):
            data_plot.delaxes(axes[i])

        # set figure super title
        figure_name = dataframe.attrs["path"].stem
        unbalance = dataframe.attrs["unbalance"].title()
        data_plot.suptitle(f"Dataset {figure_name}:    {unbalance} Unbalance")

        if not return_only:
            plt.show()
        else:
            plt.close(data_plot)

        file_name = dataframe.attrs["path"].stem + "_" + "_".join(columns)
        figure_dict = {"figure": data_plot, "file_name": file_name}
        figure_dicts.append(figure_dict)

    return figure_dicts
