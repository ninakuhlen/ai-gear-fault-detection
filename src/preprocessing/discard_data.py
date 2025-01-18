from pandas import DataFrame, Timedelta, to_timedelta
from pandas.api.types import is_integer_dtype, is_float_dtype, is_timedelta64_ns_dtype


def discard_data(
    dataframe: DataFrame,
    start: int | float | Timedelta = None,
    end: int | float | Timedelta = 50_000,
    reset_index: bool = False,
) -> DataFrame:
    """
    Discardes data from the given pandas DataFrame. The function limits 'start' and 'end'
    interpret integer values as indices and float values as time values in a unit determined
    by the DataFrame. The limits also accept pandas Timedelta.

    Args:
        dataframe (DataFrame):
            The DataFrame to discard data from.
        start (int | float | Timedelta, optional):
            The lower limit of the range to be discarded. If 'None', this attribute will be set to the start of the DataFrame. Defaults to None.
        end (int | float | Timedelta, optional):
            The upper limit of the range to be discarded. If 'None', this attribute will be set to the end of the DataFrame. Defaults to 50_000.
        reset_index (bool, optional):
            Whether to reset the indices of resulting DataFrame to start at 0. Defaults to False.

    Raises:
        AttributeError: Discarding full dataframe! Please specify either a starting value or an ending value.
        ValueError: Invalid index dtype! Only integers. floats and timedeltas are supported.

    Returns:
        DataFrame: A DataFrame containing the remaining values.
    """

    data = dataframe.copy(deep=True)

    previous_length = data.shape[0]

    if start is None and end is None:
        raise AttributeError(
            "Discarding full dataframe! Please specify either a starting value or an ending value."
        )
    elif start is None:
        start = data.index.min()
    elif end is None:
        end = data.index.max()

    print("\ndiscard_data():")

    if is_float_dtype(data.index):
        _, current_unit = data.attrs["index_type"].split("_")
        start, end = float(start), float(end)
        print(f"\tLimits interpreted as time in {current_unit}.")
    elif is_integer_dtype(data.index):
        start, end = int(start), int(end)
        print("\tLimits interpreted as indices.")
    elif is_timedelta64_ns_dtype(data.index):
        start, end = to_timedelta(start), to_timedelta(end)
        print(
            "\tLimits interpreted as time in ns and converted to pandas timedelta64[ns]."
        )
    else:
        raise ValueError(
            "Invalid index dtype! Only integers. floats and timedeltas are supported."
        )

    delete = (data.index >= start) & (data.index < end)

    data = data[~delete]

    if reset_index:
        data.reset_index(drop=True, inplace=True)

    current_length = data.shape[0]
    data.attrs["sample_size"] = f"{current_length:_}"

    print(f"\t{previous_length - current_length} rows discarded.")

    return data
