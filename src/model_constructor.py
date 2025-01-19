import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
import os
import numpy as np


def system_setup():
    """
    Displays the number of cpu and gpu cores and enables memory growth for the gpu.
    """

    n_cpu = os.cpu_count()
    print(f"Number of CPU cores:\t\t{n_cpu}")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    print(f"Number of GPUs available:\t{len(physical_devices)}")

    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


def construct_fft_net_model(
    n_hidden_layers: int,
    training_samples_dict: dict,
    l2: float = 1e-3,
    dropout: float = 0.2,
    negative_slope: float = 0.3,
) -> keras.Sequential:
    """
    Constructor for the keras Sequential model optimized for multiclass classification. Depending on the input shape, the model contains
    a backbone of convolutional layers followed by a set of fully connected hidden layers. A single
    hidden layer consists of a dense layer, a LeakyReLU activation layer and a dropout layer. If 1d
    training samples are given, no convolutions will be build in the model.

    Args:
        n_hidden_layers (int):
            The number of fully connected hidden layers.
        training_samples_dict (dict):
            A dictionary with training samples and labels to match the model input and output to.
        l2 (float, optional):
            The factor of the l2 regularization penality. It is computed as: loss = l2 * reduce_sum(square(x)).
            Defaults to 1e-3.
        dropout (float, optional): The percentage of layer inputs to set to 0. Defaults to 0.2.
        negative_slope (float, optional): The negative slope of the LeakyReLU activation function. Defaults to 0.3.

    Returns:
        keras.Sequential: A model of the described neural network optimized for multiclass classification.
    """

    input_shape = training_samples_dict["samples"].shape[1:]
    output_shape = training_samples_dict["labels"].shape[1]

    model = keras.Sequential()
    model.add(layers.Input(input_shape))

    if len(input_shape) == 1:
        model.add(layers.Dense(2048))
        dense_units = 1024
    else:
        model.add(
            layers.Conv1D(
                filters=64,
                kernel_size=3,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.L2(l2),
            )
        )
        model.add(layers.MaxPool1D(pool_size=2, strides=2, padding="valid"))

        model.add(
            layers.Conv1D(
                filters=128,
                kernel_size=3,
                activation="relu",
                padding="same",
                kernel_regularizer=regularizers.L2(l2),
            )
        )
        dense_units = 128
        model.add(layers.GlobalMaxPooling1D())

    # add the number of hidden layers
    for _ in range(n_hidden_layers):
        model.add(
            layers.Dense(
                units=dense_units, kernel_regularizer=regularizers.L2(l2)
            )
        )
        model.add(layers.LeakyReLU(negative_slope=negative_slope))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(output_shape, activation="softmax"))

    model.summary()
    return model


def compile_model(
    model: keras.Sequential,
    learning_rate: float = 1e-4,
):
    """
    Compiles the given keras Sequential model.

    Args:
        model (keras.Sequential):
            The model to compile.
        learning_rate (float):
            The learning rate used by the Adam optimizer.
    """

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),  # for multiclass classification
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def train_model(
    model: keras.Sequential,
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


def evaluate(
    model: keras.Sequential, test_samples_dict: dict, batch_size: int
) -> list[float]:
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


def predict(
    model: keras.Sequential, test_samples_dict: dict
) -> list[np.ndarray]:
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
