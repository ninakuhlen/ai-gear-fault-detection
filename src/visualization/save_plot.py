from pathlib import Path
import matplotlib.pyplot as plt


def save_plot(
    parent_path: Path,
    file_name: str,
    figure: plt.figure.Figure = None,
    format: str = "png",
    dpi: int = 300,
):

    parent_path.mkdir(mode=777, parents=True, exist_ok=True)

    if figure is None:
        figure = plt.gcf()

    file_path = parent_path / f"{file_name}.{format}"

    figure.savefig(file_path, format=format, dpi=dpi, bbox_inches="tight")
