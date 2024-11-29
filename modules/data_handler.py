import numpy as np
import pandas as pd
import yaml
import kagglehub
from pathlib import Path
from IPython.display import display
from multipledispatch import dispatch


def fetch_kaggle_dataset(handle: str):

    # move data to project data directory
    data_path = Path().cwd() / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    print("DOWNLOADING KAGGLE DATASET")
    dataset_path = Path(kagglehub.dataset_download(handle))

    for file_path in dataset_path.glob("*.csv"):
        try:
            file_path.rename(data_path / file_path.name)
            print(f"{file_path.name} successfully moved to project data folder!")
        except FileExistsError:
            print(f"{file_path.name} already exists in directory!")
            continue

    # recursive file system tree removal
    def rm_tree(path: Path):
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        path.rmdir()

    # search for .cache parent folder
    while dataset_path.name != ".cache":
        dataset_path = dataset_path.parent

    # remove .cache folder
    rm_tree(dataset_path)
    print(f"{dataset_path} fully removed!")


def split_in_training_and_test_data():
    training_data = []
    evaluation_data = []

    return (training_data, evaluation_data)


class Dataset:
    __path: Path
    __info: dict

    data: pd.DataFrame

    def __init__(self, path: Path):
        self.__path = path
        self.data = pd.read_csv(self.__path)

        try:
            __info_path = list(Path.cwd().rglob("meta_data.yaml"))[0]
            with open(__info_path, "r") as file:
                self.__info = list(yaml.safe_load_all(file))[1][self.__path.stem]

        except IndexError:
            raise IndexError("No meta_data.yaml file found!")

    def __str__(self):
        display(pd.DataFrame(self.__info, index=[0]))
        display(self.data.describe())
        return f"File name:\t{self.__path.name}\n" + f"File path:\t{self.__path}"

    def __add__(self, other):
        pd.concat(self, other)

    def reload(self):
        self.data = pd.read_csv(self.__path)

    def delete_column(self, columns: list):
        self.data.drop(columns=columns)

    def apply_threshold(
        self, threshold: any, column: str, mode: str = "le", reset_index: bool = False
    ):
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
            self.data.reset_index()

    def add_centrifugal_force(self):
        factor = self.__info["mass"] * self.__info["radius"] * 2 * np.pi / (60 * 10**6)
        self.data["CentriForce"] = self.data["Measured_RPM"].apply(
            lambda rpm: rpm * factor
        )

    @dispatch(str)
    def add_time(self, unit: str):
        if unit not in ["min", "s", "ms", "us", "ns"]:
            raise AttributeError("Invalid unit selected!")

        try:
            __info_path = list(Path.cwd().rglob("meta_data.yaml"))[0]
            with open(__info_path, "r") as file:
                sample_rate = list(yaml.safe_load_all(file))[0]["sample_rate"]
        except IndexError:
            raise IndexError("No meta_data.yaml file found!")

        if unit == "min":
            self.data[f"Time_{unit}"] = self.data.index / (60 * sample_rate)
        elif unit == "s":
            self.data[f"Time_{unit}"] = self.data.index / sample_rate
        elif unit == "ms":
            self.data[f"Time_{unit}"] = 10**3 * self.data.index / sample_rate
        elif unit == "us":
            self.data[f"Time_{unit}"] = 10**6 * self.data.index / sample_rate
        elif unit == "ns":
            self.data[f"Time_{unit}"] = 10**9 * self.data.index / sample_rate

    @dispatch()
    def add_time(self):
        try:
            __info_path = list(Path.cwd().rglob("meta_data.yaml"))[0]
            with open(__info_path, "r") as file:
                sample_rate = list(yaml.safe_load_all(file))[0]["sample_rate"]
        except IndexError:
            raise IndexError("No meta_data.yaml file found!")

        self.data[f"Time"] = pd.to_timedelta(self.data.index / sample_rate, unit="s")

    @dispatch(int)
    def resample(self, step_size: int):
        self.data = self.data.iloc[0::step_size]
        self.data.reset_index(drop=True, inplace=True)

    @dispatch(int, int)
    def resample(self, sample_size: int, random_state: int):
        self.data = self.data.sample(
            n=sample_size, random_state=random_state, ignore_index=0
        ).sort_index()
        self.data.reset_index(drop=True, inplace=True)

    @dispatch(float, int)
    def resample(self, percentage: float, random_state: int):
        self.data = self.data.sample(
            frac=percentage, random_state=random_state, ignore_index=0
        ).sort_index()
        self.data.reset_index(drop=True, inplace=True)
