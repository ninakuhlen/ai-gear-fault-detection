import numpy as np
from pandas import DataFrame


def add_centrifugal_force(dataframe: DataFrame, copy: bool = False) -> DataFrame:
    """
    Adds a column containing the centrifugal force to a DataFrame.

    Args:
        dataframe (DataFrame):
            The DataFrame to add the centrifugal force column to.
        copy (bool, optional):
            Whether to return a copy of the DataFrame. Defaults to False.

    Raises:
        AttributeError: DataFrame does not provide a 'Measured_RPM' column!

    Returns:
        DataFrame: A copy of the DataFrame, if 'copy' is True.
    """

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
