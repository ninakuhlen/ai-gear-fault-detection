from tensorflow import keras
from keras import layers, regularizers


def build_model(
    n_hidden_layers: int,
    training_samples_dict: dict,
    l2: float = 1e-3,
    dropout: float = 0.2,
    negative_slope: float = 0.3,
) -> keras.Sequential:
    """
    Constructor for the keras Sequential model optimized for binary classification. Depending on the input shape, the model contains
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

    model.add(layers.Dense(2048))
    dense_units = 1024

    # add the number of hidden layers
    for _ in range(n_hidden_layers):
        model.add(
            layers.Dense(units=dense_units, kernel_regularizer=regularizers.L2(l2))
        )
        model.add(layers.LeakyReLU(negative_slope=negative_slope))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(output_shape, activation="softmax"))  # sigmoid

    model.summary()
    return model


def compile(
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
        loss=keras.losses.BinaryCrossentropy(),  # for binary classification
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.F1Score(name="f1_score"),
        ],
    )
