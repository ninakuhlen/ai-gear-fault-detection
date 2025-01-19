import matplotlib.pyplot as plt


def plot_history(history, metrics=["loss", "accuracy", "precision", "recall"]):
    """
    Visualizes the metrics of the training.

    Args:
        history: The History-Object from Training (model.fit).
        metrics (list): List of the metrics that should be plotted. Default: loss, accuracy, precision, recall.
    """
    plt.figure(figsize=(12, 8))

    for metric in metrics:
        if metric in history.history:
            plt.plot(history.history[metric], label=f"Train {metric}")
            if f"val_{metric}" in history.history:
                plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training and Validation")
    plt.legend()
    plt.grid(True)
    plt.show()
