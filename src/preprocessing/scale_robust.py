import numpy as np
from fnmatch import fnmatch
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler


def scale_robust(
    dataframe: DataFrame, column_name: str | list[str], window_size: int = 2048
) -> DataFrame:
    """
    Robust scales a column or a set of columns. The function slides a window over the data and scales each window individually.

    Args:
        dataframe (DataFrame):
            The source DataFrame.
        column_name (str):
            The name of the column which should be scaled. This attribute accepts glob notation to specify multple columns.
        window_size (int, optional):
            The size of the window. Defaults to 2048.

    Raises:
        ValueError: No matching columns found!
        ValueError: Invalid column selected!
        ValueError: Odd window size! Please select an even window size!

    Returns:
        DataFrame: A pandas DataFrame with robust scaled column data.
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

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(5.0, 95.0),
    )

    data = dataframe.copy(deep=True)

    for column in columns:
        column_data = data[column]

        # check, if all column elements are scalars
        elements_are_scalars = all(
            column_data.apply(lambda x: np.isscalar(x) and np.isreal(x))
        )

        if not elements_are_scalars:
            raise ValueError("The column must contain only scalars!")

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

        data[column] = np.asarray(scaled_data).flatten(order="C")

    return data
