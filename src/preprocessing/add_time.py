from pandas import DataFrame


def add_time(dataframe: DataFrame, unit: str, replace_index: bool = False):
    """
    Adds a time to the dataset as a float in the specified unit as a new column.

    Args:
        unit (str):
            The selected unit. Valid values are "min", "s", "ms" "us" and "ns".
        replace_index (bool):
            Whether to overwrite the index instead of adding a new column.

    Raises:
        AttributeError: Invalid unit selected.
        IndexError: No meta.yaml file found.
    """

    time_map = {"min": 60 ** (-1), "s": 1e0, "ms": 1e3, "us": 1e6, "ns": 1e9}

    if unit not in time_map.keys():
        raise AttributeError(
            f"Invalid unit selected! Please select between:\t{time_map.keys()}"
        )

    data = dataframe.copy(deep=True)

    index_type = data.attrs["index_type"].split("_")
    current_type, current_unit = (
        index_type if len(index_type) == 2 else [index_type[0], None]
    )

    match current_type:
        case "time":
            time = time_map[unit] * data.index / time_map[current_unit]
        case "standard":
            sample_rate = data.attrs["sample_rate"]
            time = time_map[unit] * data.index / sample_rate
        case "timedelta":
            time = time_map[unit] * data.index.total_seconds()
        case _:
            raise ValueError("No matching index type!")

    if replace_index:
        data.index = time
        data.attrs["index_type"] = f"time_{unit}"
    else:
        data[f"time_{unit}"] = time

    return data
