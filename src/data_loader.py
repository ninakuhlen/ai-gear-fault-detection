import kagglehub
from pathlib import Path
from src.preprocessing import Dataset

DATASET_ADDRESS: str = "jishnukoliyadan/vibration-analysis-on-rotating-shaft"


def fetch_kaggle_dataset(handle: str):
    """
    Downloads the specified kaggle dataset and moves it to the project/data/raw directory.

    Args:
        handle (str): String representation of the kaggle dataset to download.
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


def load_datasets(parent_path: Path) -> tuple[list[Dataset], list[Dataset]]:
    """
    Loads the dataset from the specified directory and splits them into development data and evaluation data according to the .csv file names.

    Args:
        parent_path (Path): Path to the dataset directory.

    Returns:
        tuple[list[Dataset], list[Dataset]]: Two lists of Datasets. The first contains the development data and the second the evaluation data.
    """
    development_data = []
    evaluation_data = []

    print("READING TRAINING DATA")
    for file_path in parent_path.glob("*D.csv"):
        development_data.append(Dataset(file_path))
        print(f"{file_path.name} completed!")

    print("READING EVALUATION DATA")
    for file_path in parent_path.glob("*E.csv"):
        evaluation_data.append(Dataset(file_path))
        print(f"{file_path.name} completed!")

    print("READING COMPLETED")

    return development_data, evaluation_data


# TODO function to store processed data
def save_dataset(datasets: list[Dataset], uuid: str = None):
    data_path = Path().cwd() / "data" / "processed"

    if uuid:
        data_path = data_path / uuid
    data_path.mkdir(parents=True, exist_ok=True)
    pass
