import numpy as np
from pandas import DataFrame


def median(
    dataframe: DataFrame, column: str, window_size: int = 4096, stretch: bool = True
) -> np.ndarray:
    """
    Calculates the median values per window of the specified column of a given DataFrame.
    Only considers completely filled windows. Remaining data will be ignored.

    Args:
        dataframe (DataFrame):
            The DataFrame to calculate median values of.
        column (str):
            The column name.
        window_size (int, optional):
            The window size for median value calculation. Defaults to 4096.
        stretch (bool, optional):
            Whether to multiplicate every entry. The resulting arrays length will be a multiple of the window size. Defaults to True.

    Returns:
        np.ndarray: A numpy ndarray of median values.
    """

    data = dataframe
    if window_size == 0 or window_size == None:
        return data[column].median()

    # get only data from fully filled windows
    n = data.shape[0] // window_size

    median_values = []

    for i in range(n):
        start = i * window_size
        end = (i + 1) * window_size
        samples = data[column].iloc[start:end]
        median_values.append(samples.mean())

    median_values = np.asanyarray(median_values)

    if stretch:
        median_values = np.repeat(median_values, window_size)

    return median_values
