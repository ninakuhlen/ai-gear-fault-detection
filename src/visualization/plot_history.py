import matplotlib.pyplot as plt


def plot_history(
    history,
    metrics=["loss", "accuracy", "precision", "recall"],
    separated: bool = False,
) -> list:
    """
    Visualizes the metrics of the training.

    Args:
        history: The History-Object from Training (model.fit).
        metrics (list): List of the metrics that should be plotted. Default: loss, accuracy, precision, recall.
    """

    history_dict = history.history

    figure_list = []

    if separated:
        for metric in metrics:

            if metric in history_dict:
                metric_plot = plt.figure(figsize=(12, 8))
                plt.plot(history_dict[metric], label=f"Training {metric}")
                plt.plot(history_dict[f"val_{metric}"], label=f"Validation {metric}")

                plt.xlabel("Epochs")
                plt.ylabel("Value")
                plt.title(f"Training and Validation {metric.title()}")
                plt.legend()
                plt.grid(True)
                plt.show()

                file_name = metric
                figure_dict = {"figure": metric_plot, "file_name": file_name}

                figure_list.append(figure_dict)
    else:
        metric_plot = plt.figure(figsize=(12, 8))
        for metric in metrics:
            if metric in history_dict:
                plt.plot(history_dict[metric], label=f"Training {metric}")
                plt.plot(history_dict[f"val_{metric}"], label=f"Validation {metric}")

        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("Training and Validation")
        plt.legend()
        plt.grid(True)
        plt.show()

        file_name = "_".join(metrics)
        figure_dict = {"figure": metric_plot, "file_name": file_name}

        figure_list.append()

    return figure_list
