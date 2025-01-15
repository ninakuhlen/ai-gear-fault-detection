import numpy as np
from fnmatch import fnmatch
from pandas import DataFrame


def calculate_dft(
    dataframe: DataFrame,
    column_name: str | list[str],
    window_size: int = 4096,
    normalize: bool = True,
) -> DataFrame:
    """
    Fourier transforms a set of columns of a given pandas DataFrame.

    Args:
        dataframe (DataFrame): The DataFrame containing the required data.
        columns (str | list[str]):
            A list of columns or a single column name. Allows for glob notation.
        window_size (int, optional):
            The size of the window for wich the fourier coefficients are calculated. Defaults to 4096.
        normalize (bool, optional):
            Whether to normalize the fourier coefficients. Defaults to True.

    Returns:
        DataFrame: A pandas DataFrame with the fourier transformed data an their frequencies.
    """

    columns = []

    if isinstance(column_name, str):
        # search for matching columns in dataframe columns
        for column in dataframe.columns:
            if fnmatch(column, column_name):
                columns.append(column)
        if not columns:
            raise ValueError("No matching columns found!")

    elif isinstance(column_name, list):
        # check existance of columns in dataframe columns
        for column in column_name:
            if column not in dataframe.columns:
                raise ValueError(f"Invalid column '{column}' selected!")

        columns = column_name

    if window_size % 2 != 0:
        raise ValueError("Odd window size! Please select an even window size!")

    data = dataframe.copy(deep=True)

    # get only data from fully filled windows
    n = data.shape[0] // window_size

    # initialize pandas dataframe with its first column 'fft_frequency'
    fft_dataframe = DataFrame(
        np.nan, index=range(n * window_size // 2), columns=["fft_frequency"]
    )

    # calculate the frequencies
    fft_frequencies = []
    time_delta = 1 / window_size

    # only use the frequencies greater than 0
    window_fft_frequency = np.fft.rfftfreq(n=window_size, d=time_delta)[1:]

    for _ in range(n):
        fft_frequencies.append(window_fft_frequency)

    fft_dataframe["fft_frequency"] = np.asarray(fft_frequencies, dtype=float).flatten(
        order="C"
    )

    # calculate fft for each selected column
    for column in columns:

        fft_magnitudes = []

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

        # add new columns with transformed data
        fft_dataframe[f"{column}_magnitude".lower()] = np.asarray(
            fft_magnitudes, dtype=float
        ).flatten(order="C")

    # copy and edit old dataframe attributes
    fft_dataframe.attrs = data.attrs
    fft_dataframe.attrs["path"] = fft_dataframe.attrs["path"].with_stem(
        fft_dataframe.attrs["path"].stem + f"_fft"
    )
    current_length = fft_dataframe.shape[0]
    fft_dataframe.attrs["sample_size"] = f"{current_length:_}"
    fft_dataframe.attrs["sample_rate"] = window_size // 2

    return fft_dataframe
