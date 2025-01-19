from pandas import DataFrame


def random_resample(
    dataframe: DataFrame, sample_size: int | float, random_state: int = 0
):
    """
    Reduces the dataset to a fixed number of randomly selected rows. Resets the index.

    Args:
        sample_size (int | float):
            Fixed number of rows to reduce the dataset to or the percentage to reduce the datasets size to.

        random_state (int):
            Seed for random number generator.
    """

    data = dataframe.copy(deep=True)

    if isinstance(sample_size, int):
        data = data.sample(
            n=sample_size, random_state=random_state, ignore_index=0
        ).sort_index()
    elif isinstance(sample_size, float):
        data = data.sample(
            frac=sample_size, random_state=random_state, ignore_index=0
        ).sort_index()
    data.reset_index(drop=True, inplace=True)

    current_length = data.shape[0]
    data.attrs["sample_size"] = f"{current_length:_}"

    return data
