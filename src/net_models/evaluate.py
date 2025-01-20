from tensorflow.keras import Sequential


def evaluate(
    model: Sequential, test_samples_dict: dict, batch_size: int
) -> dict[str, float]:
    """
    Evaluates the performance of a given keras Sequential model.

    Args:
        model (keras.Sequential):
            The model to evaluate.
        test_samples_dict (dict):
            A sample dictionary of test data.

    Returns:
        list[float]: List of scalars for every metric used by the model.
    """

    return model.evaluate(
        test_samples_dict["samples"],
        test_samples_dict["labels"],
        batch_size=batch_size,
        verbose=1,
        return_dict=True,
    )
