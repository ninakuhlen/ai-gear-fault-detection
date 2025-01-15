import numpy as np
from pandas import DataFrame


def mean(
    dataframe: DataFrame, column: str, window_size: int = 4096, stretch: bool = True
) -> np.ndarray:
    """
    Calculates the mean values per window of the specified column of a given DataFrame.
    Only considers completely filled windows. Remaining data will be ignored.

    Args:
        dataframe (DataFrame):
            The DataFrame to calculate mean values of.
        column (str):
            The column name.
        window_size (int, optional):
            The window size for mean value calculation. Defaults to 4096.
        stretch (bool, optional):
            Whether to multiplicate every entry. The resulting arrays length will be a multiple of the window size. Defaults to True.

    Returns:
        np.ndarray: A numpy ndarray of mean values.
    """

    data = dataframe
    if window_size == 0 or window_size == None:
        return data[column].median()

    # get only data from fully filled windows
    n = data.shape[0] // window_size

    mean_values = []

    for i in range(n):
        start = i * window_size
        end = (i + 1) * window_size
        samples = data[column].iloc[start:end]
        mean_values.append(samples.mean())

    mean_values = np.asanyarray(mean_values)

    if stretch:
        mean_values = np.repeat(mean_values, window_size)

    return mean_values
