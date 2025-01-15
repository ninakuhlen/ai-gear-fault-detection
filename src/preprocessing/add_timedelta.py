from pandas import DataFrame, to_timedelta


def add_timedelta(dataframe: DataFrame, replace_index: bool = False):
    """
    Adds a timestamp to the dataset in a new column.

    Args:
        replace_index (bool):
            Whether to overwrite the index instead of adding a new column.

    Raises:
        IndexError: No meta.yaml file found.
    """

    data = dataframe.copy(deep=True)

    index_type = data.attrs["index_type"].split("_")
    current_type, current_unit = (
        index_type if len(index_type) == 2 else [index_type[0], None]
    )

    match current_type:
        case "time":
            timedelta = to_timedelta(data.index, unit=current_unit)
        case "standard":
            sample_rate = data.attrs["sample_rate"]
            timedelta = to_timedelta(data.index / sample_rate, unit="s")
        case "timedelta":
            timedelta = data.index
            print("Index is TimeDelta already.")
        case _:
            raise ValueError("No matching index type!")

    if replace_index:
        data.index = timedelta
        data.attrs["index_type"] = "timedelta"
    else:
        data["timedelta"] = timedelta

    return data
