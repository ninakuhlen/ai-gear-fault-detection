import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from IPython.display import display
from multipledispatch import dispatch


class Dataset:
    """
    Wrapper class for pandas DataFrame with methods for data preprocessing.

    Returns:
        Dataset: An instance of the Dataset class.
    """

    __path: Path
    __info: dict

    data: pd.DataFrame

    def __init__(self, path: Path):
        """
        Constructor of a Dataset object.

        Args:
            path (Path): File path to the dataset .csv file.

        Raises:
            IndexError: No meta_data.yaml file found.
        """
        self.__path = path
        self.data = pd.read_csv(self.__path)

        try:
            info_path = list(Path.cwd().rglob("meta_data.yaml"))[0]
            with open(info_path, "r") as file:
                self.__info = list(yaml.safe_load_all(file))[1][self.__path.stem]

        except IndexError:
            raise IndexError("No meta_data.yaml file found!")

    def __str__(self):
        """
        Dunder method of the 'str()' operation. A Dataset will display its metadata and a pandas DataFrame description.

        Returns:
            str: File name and file path.
        """
        display(pd.DataFrame(self.__info, index=[0]))
        display(self.data.describe())
        return f"File name:\t{self.__path.name}\n" + f"File path:\t{self.__path}"

    def __add__(self, other):
        """
        Dunder method of the '+' operation. Two datasets are concatenated in the order of addition.

        Args:
            other (Dataset): The other Dataset.
        """
        pd.concat(self, other)

    def reload(self):
        """
        Reloads the raw data into the Dataset.
        """
        self.data = pd.read_csv(self.__path)

    def delete_columns(self, columns: list):
        """
        Deletes the selected list of columns.

        Args:
            columns (list): List of columns to be deleted.
        """
        self.data.drop(columns=columns, inplace=True)

    def delete_all_added_columns(self):
        """
        Deletes all columns except those of the raw dataset. All applied filters, threshold values and changes to the index are retained.

        Raises:
            IndexError: No meta_data.yaml file found.
        """
        try:
            info_path = list(Path.cwd().rglob("meta_data.yaml"))[0]
            with open(info_path, "r") as file:
                columns = list(yaml.safe_load_all(file))[0]["columns"]

        except IndexError:
            raise IndexError("No meta_data.yaml file found!")

        self.delete_columns(columns=self.data.columns.difference(columns))

    def apply_threshold(
        self, threshold: any, column: str, mode: str = "le", reset_index: bool = False
    ):
        """
        Filters the dataset by comparing column values against a threshold value.

        Args:
            threshold (any): The threshold according to which the data is filtered.
            column (str): The column to which the threshold is applied.
            mode (str, optional): The comparison operator for thresholding. Defaults to "le".
            reset_index (bool, optional): Whether the index is reset to the start value 0 or not. Defaults to False.

        Raises:
            AttributeError: Invalid thresholding mode selected.
        """
        if mode not in ["eq", "le", "ge", "lt", "gt"]:
            raise AttributeError("Invalid thresholding mode selected!")

        if mode == "eq":
            self.data.drop(
                self.data.index[self.data[column] == threshold], inplace=True
            )
        elif mode == "le":
            self.data.drop(
                self.data.index[self.data[column] <= threshold], inplace=True
            )
        elif mode == "ge":
            self.data.drop(
                self.data.index[self.data[column] >= threshold], inplace=True
            )
        elif mode == "lt":
            self.data.drop(self.data.index[self.data[column] < threshold], inplace=True)
        elif mode == "gt":
            self.data.drop(self.data.index[self.data[column] > threshold], inplace=True)

        if reset_index:
            self.data.reset_index(drop=True, inplace=True)

    def add_centrifugal_force(self):
        factor = self.__info["mass"] * self.__info["radius"] * 2 * np.pi / (60 * 10**6)
        self.data["CentriForce"] = self.data["Measured_RPM"].apply(
            lambda rpm: rpm * factor
        )

    @dispatch(str, bool)
    def add_time(self, unit: str, replace_index: bool):
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

        try:
            __info_path = list(Path.cwd().rglob("meta_data.yaml"))[0]
            with open(__info_path, "r") as file:
                sample_rate = list(yaml.safe_load_all(file))[0]["sample_rate"]
        except IndexError:
            raise IndexError("No meta_data.yaml file found!")

        if unit == "min":
            time = self.data.index / (60 * sample_rate)
        elif unit == "s":
            time = self.data.index / sample_rate
        elif unit == "ms":
            time = 10**3 * self.data.index / sample_rate
        elif unit == "us":
            time = 10**6 * self.data.index / sample_rate
        elif unit == "ns":
            time = 10**9 * self.data.index / sample_rate

        if replace_index:
            self.data.index = time
        else:
            self.data[f"Time_{unit}"] = time

    @dispatch(bool)
    def add_time(self, replace_index: bool):
        """
        Adds a timestamp to the dataset in a new column.

        Args:
            replace_index (bool): Overwrites the index instead of adding a new column.

        Raises:
            IndexError: No meta.yaml file found.
        """
        try:
            __info_path = list(Path.cwd().rglob("meta_data.yaml"))[0]
            with open(__info_path, "r") as file:
                sample_rate = list(yaml.safe_load_all(file))[0]["sample_rate"]
        except IndexError:
            raise IndexError("No meta_data.yaml file found!")

        if replace_index:
            self.data.index = pd.to_timedelta(self.data.index / sample_rate, unit="s")
        else:
            self.data[f"Time"] = pd.to_timedelta(
                self.data.index / sample_rate, unit="s"
            )

    @dispatch(int)
    def resample(self, step_size: int):
        """
        Reduces the dataset by selecting every nth row.

        Args:
            step_size (int): The step size with which the dataset is reduced.
        """
        self.data = self.data.iloc[0::step_size]
        self.data.reset_index(drop=True, inplace=True)

    @dispatch(int, int)
    def resample(self, sample_size: int, random_state: int):
        """
        Reduces the dataset to a fixed number of randomly selected rows.

        Args:
            sample_size (int): Fixed number of rows to reduce the dataset to.
            random_state (int): Seed for random number generator.
        """
        self.data = self.data.sample(
            n=sample_size, random_state=random_state, ignore_index=0
        ).sort_index()
        self.data.reset_index(drop=True, inplace=True)

    @dispatch(float, int)
    def resample(self, percentage: float, random_state: int):
        """
        Reduces the data set randomly to a percentage of the original dataset.

        Args:
            percentage (float): Percentage of the original dataset to reduce the dataset to.
            random_state (int): Seed for random number generator.
        """
        self.data = self.data.sample(
            frac=percentage, random_state=random_state, ignore_index=0
        ).sort_index()
        self.data.reset_index(drop=True, inplace=True)
