from pandas import DataFrame


def apply_threshold(
    dataframe: DataFrame,
    threshold: any,
    column: str,
    mode: str = "le",
    copy: bool = False,
    reset_index: bool = False,
) -> DataFrame:
    """
    Filters the dataset by comparing column values against a threshold value.

    Args:
        dataframe(DataFrame):
            The dataset to apply the thresholding to.
        threshold (any):
            The threshold according to which the data is filtered.
        column (str):
            The column to which the threshold is applied.
        mode (str, optional):
            The comparison operator for thresholding. Defaults to "le".
        copy (bool, optional):
            If True returns a copy of the filtered DataFrame. Defaults to False.
        reset_index (bool, optional):
            Whether the index is reset to the start value 0 or not. Defaults to False.

    Raises:
        AttributeError: Invalid thresholding mode selected.

    Returns:
        DataFrame: The dataset copy, if 'copy' is True.
    """

    if mode not in ["eq", "le", "ge", "lt", "gt"]:
        raise AttributeError("Invalid thresholding mode selected!")

    if copy:
        data = dataframe.copy(deep=True)
    else:
        data = dataframe

    previous_length = data.shape[0]

    if mode == "eq":
        data.drop(data.index[data[column] == threshold], inplace=True)
    elif mode == "le":
        data.drop(data.index[data[column] <= threshold], inplace=True)
    elif mode == "ge":
        data.drop(data.index[data[column] >= threshold], inplace=True)
    elif mode == "lt":
        data.drop(data.index[data[column] < threshold], inplace=True)
    elif mode == "gt":
        data.drop(data.index[data[column] > threshold], inplace=True)

    current_length = data.shape[0]
    data.attrs["sample_size"] = f"{current_length:_}"

    print("\napply_threshold():")
    print(f"\tInput:\t{dataframe.attrs['path'].stem}")
    print(f"\t{previous_length - current_length} rows discarded.")

    if reset_index:
        data.reset_index(drop=True, inplace=True)

    if copy:
        return data
