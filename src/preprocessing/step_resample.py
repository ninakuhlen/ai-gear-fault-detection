from pandas import DataFrame


def step_resample(dataframe: DataFrame, step_size: int):
    """
    Reduces the dataset by selecting every nth row. Resets the index.

    Args:
        step_size (int):
            The step size with which the dataset is reduced.
    """

    data = dataframe.copy(deep=True)
    data = data.iloc[0::step_size]
    data.reset_index(drop=True, inplace=True)

    current_length = data.shape[0]
    data.attrs["sample_size"] = f"{current_length:_}"

    return data
