import kagglehub
from copy import deepcopy
from pandas import DataFrame, read_csv
from pathlib import Path
from yaml import safe_load, safe_dump
from tensorflow import keras

DATASET_ADDRESS: str = "jishnukoliyadan/vibration-analysis-on-rotating-shaft"


def fetch_kaggle_dataset(handle: str):
    """
    Downloads the specified kaggle dataset and moves it to the project/data/raw directory.

    Args:
        handle (str):
            String representation of the kaggle dataset to download.
    """

    # move data to project data directory
    data_path = Path().cwd() / "data" / "raw"
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
    def rm_tree(parent_path: Path):
        for child in parent_path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        parent_path.rmdir()

    # search for .cache parent folder
    while dataset_path.name != ".cache":
        dataset_path = dataset_path.parent

    # remove .cache folder
    rm_tree(dataset_path)
    print(f"{dataset_path} fully removed!")


def load_all_datasets(parent_path: Path) -> tuple[list[DataFrame], list[DataFrame]]:
    """
    Loads the dataset from the specified directory and splits them into development data and evaluation data according to the .csv file names.

    Args:
        parent_path (Path):
            Path to the dataset directory.

    Returns:
        tuple[list[Dataset], list[Dataset]]: Two lists of Datasets. The first contains the development data and the second the evaluation data.
    """
    development_data = []
    evaluation_data = []

    print("load_all_datasets():")

    print("\tReading development data ...")
    for file_path in parent_path.glob("*D*.csv"):
        development_data.append(load_dataset(file_path))
        print(f"\t{file_path.name} successfully loaded.")

    print("\tReading evaluation data ...")
    for file_path in parent_path.glob("*E*.csv"):
        evaluation_data.append(load_dataset(file_path))
        print(f"\t{file_path.name} successfully loaded.")

    print("\tReading completed.\n")

    return development_data, evaluation_data


def load_dataset(path: Path) -> DataFrame:
    """
    Loads a single pandas DataFrame from a given path.

    Args:
        path (Path): Path to the csv file.

    Raises:
        IndexError: No meta_data.yaml file found in parent directory!

    Returns:
        DataFrame: Pandas DataFrame created from the csv file.
    """

    dataframe = read_csv(path)

    try:
        file = list(path.parent.rglob("meta_data.yaml"))[0]
        yaml_data = safe_load(file.read_text())
        dataframe.attrs = yaml_data[path.stem]
        dataframe.attrs["path"] = path
        if "index_type" not in dataframe.attrs.keys():
            dataframe.attrs["index_type"] = "index"
        dataframe.attrs["sample_size"] = dataframe.shape[0]

    except IndexError:
        raise IndexError("No meta_data.yaml file found in parent directory!")
    return dataframe


def save_dataset(dataframe: DataFrame, uuid: str):
    """
    Saves a pandas DataFrame to a csv file and stores its DataFrame.attrs to a yaml file.

    Args:
        dataframe (DataFrame):
            The DataFrame instance to be saved.
        uuid (str):
            The foldername universally unique identifier to be used as directory name.
    """

    parent_path = Path().cwd() / "data" / "processed" / f"{uuid}"

    parent_path.mkdir(mode=777, parents=True, exist_ok=True)

    csv_path = parent_path / dataframe.attrs["path"].name
    dataframe.attrs["path"] = csv_path
    dataframe.to_csv(csv_path, index=False)

    yaml_path = parent_path / "meta_data.yaml"

    yaml_data = deepcopy(dataframe.attrs)
    yaml_data["path"] = str(yaml_data["path"])
    yaml_data = {dataframe.attrs["path"].stem: yaml_data}

    append_to_yaml(yaml_path, yaml_data)

    print(f"{dataframe.attrs['path'].name} successfully saved.")


def append_to_yaml(file: Path, data: dict):
    """
    Creates or adds to a yaml file.

    Args:
        file (Path):
            Path to the yaml file.
        data (dict):
            A dictionary with the data to be written to the yaml file.
    """

    if not file.exists():
        file.write_text(safe_dump(data))
    else:
        yaml_data = safe_load(file.read_text())
        yaml_data = yaml_data | data
        file.write_text(safe_dump(yaml_data))


def save_figure(
    parent_path: Path,
    figure_dict: dict,
    format: str = "png",
    dpi: int = 300,
):

    parent_path.mkdir(mode=777, parents=True, exist_ok=True)

    similar_files = list(parent_path.rglob(f"{figure_dict['file_name']}*.{format}"))

    new_file_name = figure_dict["file_name"] + "_" + chr(97 + len(similar_files))

    file_path = parent_path / f"{new_file_name}.{format}"

    figure_dict["figure"].savefig(
        file_path, format=format, dpi=dpi, bbox_inches="tight"
    )


def save_model_hyperparameters(
    parent_path: Path,
    model,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    validation_split: float,
):

    dropout_rates = [
        layer.rate for layer in model.layers if isinstance(layer, keras.layers.Dropout)
    ]
    l2_rates = [
        layer.kernel_regularizer.l2
        for layer in model.layers
        if hasattr(layer, "kernel_regularizer")
        and isinstance(layer.kernel_regularizer, keras.regularizers.L2)
    ]

    n_hidden_layers = len(model.layers) - 2

    hyperparameters = {
        "model": {
            "learning_rate": learning_rate,
            "dropout_rates": dropout_rates,
            "l2_rates": l2_rates,
            "n_hidden_layers": n_hidden_layers,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": validation_split,
        },
    }

    yaml_path = parent_path / "hyperparameters.yaml"
    append_to_yaml(file=yaml_path, data=hyperparameters)
