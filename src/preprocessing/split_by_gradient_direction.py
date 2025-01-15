import numpy as np
from pandas import DataFrame


def split_by_gradient_direction(
    dataframe: DataFrame,
    column: str,
    periods: int = 1,
    sign: int = -1,
    min_length: int = 50_000,
    reset_index: bool = False,
) -> list[DataFrame]:
    """
    Analyses the data of a specified pandas DataFrame column for gradient directions and splits the dataset accordingly.

    Args:
        dataframe (DataFrame):
            The DataFrame to split by gradient direction.
        column (str):
            The column to analyze for gradient direction.
        periods (int, optional):
            Periods to shift for calculating difference. Defaults to 1.
        sign (int, optional):
            The direction of the gradient: 1 means ascending and -1 descending. Defaults to -1.
        min_length (int, optional):
            A minimum size of dataframe subsets. Subsets below this length will be discarded. Defaults to 50_000.
        reset_index (bool, optional):
            Whether to reset the indices of the created subsets to start at 0. Defaults to False.

    Raises:
        AttributeError: Invalid sign selected! Please specify 1 for raising and -1 for falling gradient.

    Returns:
        list[DataFrame]: The subsets generated.
    """

    if sign not in [1, -1]:
        raise AttributeError(
            "Invalid sign selected! Please specify 1 for raising and -1 for falling gradient."
        )

    data = dataframe[column].copy(deep=True)

    differences = data.diff(periods=periods)
    indices = differences.index[np.sign(differences.values) == -1].tolist()
    indices.append(data.index.max())

    dataframes = []

    start = 0
    for subset_number, index in enumerate(indices):
        end = index
        subset = dataframe[start:end]

        length = subset.shape[0]

        if length >= min_length:

            # copy dataframe info to subset
            subset.attrs = dataframe.attrs

            # add number to subset path stem
            subset.attrs["path"] = subset.attrs["path"].with_stem(
                subset.attrs["path"].stem + f"_{subset_number}"
            )

            # match the sample size in subset info
            current_length = subset.shape[0]
            subset.attrs["sample_size"] = f"{current_length:_}"

            if reset_index:
                subset.reset_index(drop=True, inplace=True)

            dataframes.append(subset)

        start = end

    return dataframes
