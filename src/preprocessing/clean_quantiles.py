import numpy as np
from pandas import DataFrame, Series
from fnmatch import fnmatch


def clean_quantiles(
    dataframe: DataFrame,
    column_name: str | list[str],
    quantiles: tuple[float] = (0.05, 0.95),
    window_size: int = 2048,
    discard: bool = False,
) -> DataFrame:

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

    def adjust_outliers(sample: Series, quantiles: tuple[float], discard: bool):
        lower_quantile = np.quantile(sample, quantiles[0])
        upper_quantile = np.quantile(sample, quantiles[1])

        outliers = (sample >= upper_quantile) | (sample <= lower_quantile)
        n_outliers = len(outliers[outliers == True])

        if discard:
            sample[(sample >= upper_quantile) | (sample <= lower_quantile)] = np.nan
        else:
            sample[sample >= upper_quantile] = upper_quantile
            sample[sample <= lower_quantile] = lower_quantile

        return sample, n_outliers

    data = dataframe.copy(deep=True)

    old_length = len(data)

    print("\nclean_quantiles():")

    for column in columns:
        column_data = data[column]

        # get only data from fully filled windows
        n = data.shape[0] // window_size

        adjusted_samples_list = []
        n_outliers_total = 0

        for i in range(n):

            start = i * window_size
            end = (i + 1) * window_size
            sample = column_data.iloc[start:end]

            adjusted_sample, n_outliers = adjust_outliers(
                sample=sample, quantiles=quantiles, discard=discard
            )
            adjusted_samples_list.append(adjusted_sample)
            n_outliers_total += n_outliers

        # predict the final not fully filled window
        sample = column_data.iloc[end:]
        if len(sample):
            adjusted_sample = adjust_outliers(
                sample=sample, quantiles=quantiles, discard=discard
            )
            adjusted_samples_list.append(adjusted_sample)

        data[column] = np.asarray(adjusted_samples_list).flatten(order="C")

        print(f"\t{n_outliers_total:_} values adjusted/removed in column '{column}'!")

    if discard:
        data = data.dropna()

    new_length = len(data)

    print(f"\t{(old_length - new_length):_} rows discarded from dataframe!")

    return data
