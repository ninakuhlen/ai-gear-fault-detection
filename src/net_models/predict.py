import numpy as np
from tensorflow.keras import Sequential


def predict(model: Sequential, test_samples_dict: dict) -> list[np.ndarray]:
    """
    Generates predictions for the input samples.

    Args:
        model (keras.Sequential):
            The model to use for prediction.
        test_samples_dict (dict):
            A sample dictionary of test data.

    Returns:
        list[np.ndarray]: _description_
    """
    prediction = model.predict(test_samples_dict["samples"])
    true_labels = test_samples_dict["labels"].argmax(axis=1)
    predicted_labels = prediction.argmax(axis=1)

    print("\nMODEL PREDICTIONS\n")
    print(true_labels)
    print(predicted_labels)

    return true_labels, predicted_labels
