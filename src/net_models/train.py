from tensorflow.keras import Sequential, callbacks


def train(
    model: Sequential,
    samples_dict: dict,
    epochs: int,
    batch_size: int,
    validation_split: float = 0.1,
    use_early_stopping: bool = True,
) -> callbacks.History:
    """
    Trains the given keras Sequential model for multiclass classification.

    Args:
        model (keras.Sequential):
            The model to train.
        samples_dict (dict):
            The samples dictionary containing:
                - "samples": The input training data (features).
                - "labels": One-hot encoded labels for multiclass classification.
                - "class_weights": Class weights for handling imbalanced datasets.
        epochs (int):
            The number of epochs for training.
        batch_size (int):
            The training batch size.
        validation_split (float, optional):
            The percentage of training samples to use as validation data. Defaults to 0.1.
        use_early_stopping (bool, optional):
            Whether to use an EarlyStopping callback function. Defaults to True.

    Returns:
        callbacks.History: A keras history object containing the training metrics.
    """
    callback_list = []
    if use_early_stopping:
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        callback_list.append(early_stopping)

    return model.fit(
        x=samples_dict["samples"],
        y=samples_dict["labels"],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callback_list,
        class_weight=samples_dict["class_weights"],
    )
