import numpy as np
from fnmatch import fnmatch
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression


def clean_outliers(
    dataframe: DataFrame,
    column_name: str | list[str],
    window_size: int = None,
    std_multiplier: float = 2,
    discard: bool = False,
) -> DataFrame:
    """
    Perform linear regression using the DataFrame index as the independent variable and a specified column as the dependent variable.
    This may be performed fully or per window for each data column. Outliers are corrected to values of the regression line by default.

    Args:
        dataframe (DataFrame):
            The input DataFrame.
        column_name (str | list[str]):
            The name(s) of the column(s) to be used as the dependent variable.
        window_size (int, optional):
            The size of the window. If 'None' the full columns are used for a single regression. Defaults to None.
        std_multiplier (float, optional):
            The number of standard deviations above and below which deviations are considered outliers. Defaults to 2.
        discard (bool, optional):
            Whether or not the outliers and the corresponding data points of other columns should be discarded. Defaults to False.

    Raises:
        ValueError: No matching columns found!
        ValueError: Invalid column selected!
        ValueError: Odd window size! Please select an even window size!

    Returns:
        DataFrame: A DataFrame without outliers.
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

    if window_size is not None and window_size % 2 != 0:
        raise ValueError("Odd window size! Please select an even window size!")

    def linear_regression(data: Series) -> tuple[np.ndarray, np.ndarray]:
        x_values = data.index.values.reshape(-1, 1)
        y_values = data.values.reshape(-1, 1)

        # create and train the model
        model = LinearRegression()
        model.fit(x_values, y_values)

        # Make predictions
        y_predictions = model.predict(x_values)

        return y_values.flatten(), y_predictions.flatten()

    data = dataframe.copy(deep=True)

    if discard:
        discard_indices_list = []

    print("\nclean_outliers():")

    for column in columns:

        column_data = data[column]

        if not window_size:
            y_values, y_predictions = linear_regression(column_data)
        else:
            # get only data from fully filled windows
            n = data.shape[0] // window_size
            y_values_list = []
            y_predictions_list = []

            # predict the fully filled windows
            for i in range(n):
                start = i * window_size
                end = (i + 1) * window_size
                sample = column_data.iloc[start:end]
                y_values, y_predictions = linear_regression(sample)
                y_values_list.append(y_values)
                y_predictions_list.append(y_predictions)

            if not discard:
                # predict the final not fully filled window
                sample = column_data.iloc[end:]
                y_values, y_predictions = linear_regression(sample)
                y_values_list.append(y_values)
                y_predictions_list.append(y_predictions)

            # merge all values to single ndarrays
            y_values = np.vstack(y_values_list).flatten(order="C")
            y_predictions = np.vstack(y_predictions_list).flatten(order="C")
            print(y_predictions.shape)

        # calculate residuals and standard deviation
        residuals = y_values - y_predictions
        std_deviation = np.std(residuals)

        # identify outliers
        threshold = std_multiplier * std_deviation
        outliers = np.abs(residuals) > threshold

        n_outliers = len(outliers[outliers == True])

        print(f"{n_outliers} values adjusted/discarded from column '{column}'!")

        if discard:
            # collect row indices to discard
            discard_indices = np.where(outliers == True)[0]
            discard_indices_list.append(discard_indices)
        else:
            # adjust outliers with points on the regression line
            data[column] = np.where(outliers, y_predictions, y_values)

    if discard:
        discard_indices = np.vstack(discard_indices_list).flatten(order="C")
        data = data.drop(data.index[discard_indices])

    return data
