import numpy as np
from pandas import DataFrame, to_timedelta, Timedelta
from pandas.api.types import is_integer_dtype, is_float_dtype, is_timedelta64_ns_dtype
from sklearn.preprocessing import RobustScaler


def apply_threshold(
    dataframe: DataFrame,
    threshold: any,
    column: str,
    mode: str = "le",
    copy: bool = False,
    reset_index: bool = False,
):
    """
    Filters the dataset by comparing column values against a threshold value.

    Args:
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

    print(f"{previous_length - current_length} rows discarded.")

    if reset_index:
        data.reset_index(drop=True, inplace=True)

    if copy:
        return data

def split_by_gradient_direction(dataframe: DataFrame, column: str, periods: int = 1, sign: int = -1, min_length: int = 50_000, reset_index: bool =False):

    if sign not in [1, -1]:
        raise AttributeError("Invalid sign selected! Please specify 1 for raising and -1 for falling gradient.")

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
            subset.attrs["path"] = subset.attrs["path"].with_stem(subset.attrs["path"].stem + f"_{subset_number}")

            # match the sample size in subset info
            current_length = subset.shape[0]
            subset.attrs["sample_size"] = f"{current_length:_}"

            if reset_index:
                subset.reset_index(drop=True, inplace=True)

            dataframes.append(subset)
            
        start = end

    return dataframes

def discard_data(dataframe: DataFrame, start: int|float|Timedelta = None, end: int| float|Timedelta = 50_000, reset_index: bool = False):

    data = dataframe.copy(deep=True)

    previous_length = data.shape[0]

    if start is None and end is None:
        raise AttributeError("Discarding full dataframe! Please specify either a starting value or an ending value.")
    elif start is None:
        start = data.index.min()
    elif end is None:
        end = data.index.max()

    if is_float_dtype(data.index):
        _, current_unit = data.attrs["index_type"].split("_")
        start, end = float(start), float(end)
        print(f"Limits interpreted as time in{current_unit}")
    elif is_integer_dtype(data.index):
        start, end = int(start), int(end)
        print("Limits interpreted as integers.")
    elif is_timedelta64_ns_dtype(data.index):
        start, end = to_timedelta(start), to_timedelta(end)
        print("Limits interpreted as time in ns and converted to pandas timedelta64[ns].")
    else:
        raise ValueError("Invalid index dtype! Only integers. floats and timedeltas are supported.")
    
    delete = (data.index >= start) & (data.index < end)

    data = data[~delete]

    if reset_index:
        data.reset_index(drop=True, inplace=True)

    current_length = data.shape[0]
    data.attrs["sample_size"] = f"{current_length:_}"

    print(f"{previous_length - current_length} rows discarded.")

    return data


def add_centrifugal_force(dataframe: DataFrame, copy: bool = False):
    if "Measured_RPM" not in dataframe.columns:
        raise AttributeError("DataFrame does not provide a 'Measured_RPM' column!")

    if copy:
        data = dataframe.copy(deep=True)
    else:
        data = dataframe

    factor = data.attrs["mass"] * data.attrs["radius"] * 2 * np.pi / (60 * 10**6)
    data["force"] = data["Measured_RPM"].apply(lambda rpm: rpm * factor)

    if copy:
        return data


def add_time(dataframe: DataFrame, unit: str, replace_index: bool = False):
    """
    Adds a time to the dataset as a float in the specified unit as a new column.

    Args:
        unit (str): The selected unit. Valid values are "min", "s", "ms" "us" and "ns".
        replace_index (bool): Overwrites the index instead of adding a new column

    Raises:
        AttributeError: Invalid unit selected.
        IndexError: No meta.yaml file found.
    """

    time_map = {
        "min": 60**(-1),
        "s": 1e0,
        "ms": 1e3,
        "us": 1e6,
        "ns": 1e9
    }

    if unit not in time_map.keys():
        raise AttributeError(f"Invalid unit selected! Please select between:\t{time_map.keys()}")
    
    data = dataframe.copy(deep=True)

    index_type = data.attrs["index_type"].split("_")
    current_type, current_unit = index_type if len(index_type) == 2 else [index_type[0], None]
   
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


def add_timedelta(dataframe: DataFrame, replace_index: bool = False):
    """
    Adds a timestamp to the dataset in a new column.

    Args:
        replace_index (bool): Overwrites the index instead of adding a new column.

    Raises:
        IndexError: No meta.yaml file found.
    """

    data = dataframe.copy(deep=True)

    index_type = data.attrs["index_type"].split("_")
    current_type, current_unit = index_type if len(index_type) == 2 else [index_type[0], None]

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


def step_resample(dataframe: DataFrame, step_size: int):
    """
    Reduces the dataset by selecting every nth row. Resets the index.

    Args:
        step_size (int): The step size with which the dataset is reduced.
    """

    data = dataframe.copy(deep=True)
    data = data.iloc[0::step_size]
    data.reset_index(drop=True, inplace=True)


    current_length = data.shape[0]
    data.attrs["sample_size"] = f"{current_length:_}"

    return data


def random_resample(
    dataframe: DataFrame, sample_size: int | float, random_state: int = 0
):
    """
    Reduces the dataset to a fixed number of randomly selected rows. Resets the index

    Args:
        sample_size (int): Fixed number of rows to reduce the dataset to.
        random_state (int): Seed for random number generator.
    """

    data = dataframe.copy(deep=True)

    match type(sample_size):
        case "int":
            data = data.sample(
                n=sample_size, random_state=random_state, ignore_index=0
            ).sort_index()
        case "float":
            data = data.sample(
                frac=sample_size, random_state=random_state, ignore_index=0
            ).sort_index()
    data.reset_index(drop=True, inplace=True)

    current_length = data.shape[0]
    data.attrs["sample_size"] = f"{current_length:_}"

    return data


def median(
    dataframe: DataFrame, column: str, window_size: int = 4096, stretch: bool = True
):

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


def mean(
    dataframe: DataFrame, column: str, window_size: int = 4096, stretch: bool = True
):

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


def calculate_fft_magnitudes(
    dataframe: DataFrame, column: str, window_size: int = 4096, normalize: bool = True
):
    if column not in dataframe.columns:
        raise AttributeError("Invalid column selected!")
    if window_size % 2 != 0:
        raise AttributeError("Please select an even window size!")

    data = dataframe.copy(deep=True)

    # get only data from fully filled windows
    n = data.shape[0] // window_size

    fft_magnitudes = []
    fft_frequencies = []

    for i in range(n):
        start = i * window_size
        end = (i + 1) * window_size
        samples = data[column].iloc[start:end]

        if normalize:
            # calculate the normalized magnitudes of the fourier coefficients
            fft_data = 2 * np.abs(np.fft.rfft(samples)) / window_size
            # only use the first window_size / 2 values
            fft_magnitudes.append(fft_data[1:])
        else:
            # calculate the magnitudes of the fourier coefficients
            fft_data = np.abs(np.fft.rfft(samples))
            # only use the first window_size / 2 values
            fft_magnitudes.append(fft_data[1:])

        # calculate the frequencies to the magnitudes
        time_delta = 1 / window_size
        fft_frequency = np.fft.rfftfreq(n=window_size, d=time_delta)
        # only use the first window_size / 2 values
        fft_frequencies.append(fft_frequency[1:])

    fft_magnitudes = np.asarray(fft_magnitudes, dtype=float).flatten(order="C")
    fft_frequencies = np.asarray(fft_frequencies, dtype=float).flatten(order="C")

    # create new dataframe with transformed data
    fft_dataframe = DataFrame({"fft_frequency": fft_frequencies, "fft_magnitude": fft_magnitudes})
    fft_dataframe.attrs = data.attrs

    # add '_fft' to file name
    fft_dataframe.attrs["path"] = fft_dataframe.attrs["path"].with_stem(fft_dataframe.attrs["path"].stem + f"_fft")

    current_length = fft_dataframe.shape[0]
    fft_dataframe.attrs["sample_size"] = f"{current_length:_}"

    fft_dataframe.attrs["sample_rate"] = window_size // 2
    
    return fft_dataframe


def scale_robust(dataframe: DataFrame, column: str, window_size: int = 2048):
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(5.0, 95.0),
    )

    data = dataframe.copy(deep=True)

    column_data = data[column]

    # check, if all column elements are scalars
    elements_are_scalars = all(
        column_data.apply(lambda x: np.isscalar(x) and np.isreal(x))
    )

    if elements_are_scalars:

        # get only data from fully filled windows
        n = data.shape[0] // window_size

        scaled_data = []

        for i in range(n):

            start = i * window_size
            end = (i + 1) * window_size
            sample = column_data.iloc[start:end]

            scaled_sample = scaler.fit_transform(sample.values.reshape(-1, 1)).flatten(
                order="C"
            )
            scaled_data.append(scaled_sample)

    else:
        raise ValueError("The column must contain only scalars!")

    data[column] = np.asarray(scaled_data).flatten(order="C")

    return data
