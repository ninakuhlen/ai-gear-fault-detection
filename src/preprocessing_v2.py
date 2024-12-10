import numpy as np
from pandas import DataFrame, to_timedelta
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

    if reset_index:
        data.reset_index(drop=True, inplace=True)

    if copy:
        return data


def add_centrifugal_force(dataframe: DataFrame, label: str, copy: bool = False):
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


def add_time(dataframe: DataFrame, unit: str, copy: bool, replace_index: bool):
    """
    Adds a time to the dataset as a float in the specified unit as a new column.

    Args:
        unit (str): The selected unit. Valid values are "min", "s", "ms" "us" and "ns".
        replace_index (bool): Overwrites the index instead of adding a new column

    Raises:
        AttributeError: Invalid unit selected.
        IndexError: No meta.yaml file found.
    """
    if unit not in ["min", "s", "ms", "us", "ns"]:
        raise AttributeError("Invalid unit selected!")

    if copy:
        data = dataframe.copy(deep=True)
    else:
        data = dataframe

    sample_rate = data.attrs["sample_rate"]

    if unit == "min":
        time = data.index / (60 * sample_rate)
    elif unit == "s":
        time = data.index / sample_rate
    elif unit == "ms":
        time = 10**3 * data.index / sample_rate
    elif unit == "us":
        time = 10**6 * data.index / sample_rate
    elif unit == "ns":
        time = 10**9 * data.index / sample_rate

    if replace_index:
        data.index = time
    else:
        data[f"Time_{unit}"] = time

    if copy:
        return data


def add_timedelta(dataframe: DataFrame, copy: bool, replace_index: bool):
    """
    Adds a timestamp to the dataset in a new column.

    Args:
        replace_index (bool): Overwrites the index instead of adding a new column.

    Raises:
        IndexError: No meta.yaml file found.
    """

    if copy:
        data = dataframe.copy(deep=True)
    else:
        data = dataframe

    sample_rate = data.attrs["sample_rate"]

    if replace_index:
        data.index = to_timedelta(data.index / sample_rate, unit="s")
    else:
        data[f"Time"] = to_timedelta(data.index / sample_rate, unit="s")

    if copy:
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
    return data


def random_resample(
    dataframe: DataFrame, sample_size: int | float, random_state: int = 0
):
    """
    Reduces the dataset to a fixed number of randomly selected rows.

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

    data = dataframe

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
            fft_magnitudes.append(fft_data[:-1])
        else:
            # calculate the magnitudes of the fourier coefficients
            fft_data = np.abs(np.fft.rfft(samples))
            # only use the first window_size / 2 values
            fft_magnitudes.append(fft_data[:-1])

        # calculate the frequencies to the magnitudes
        time_delta = 1 / window_size
        fft_frequency = np.fft.rfftfreq(n=window_size, d=time_delta)
        # only use the first window_size / 2 values
        fft_frequencies.append(fft_frequency[:-1])

    fft_magnitudes = np.asarray(fft_magnitudes, dtype=float).flatten(order="C")
    fft_frequencies = np.asarray(fft_frequencies, dtype=float).flatten(order="C")

    return fft_frequencies, fft_magnitudes


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
