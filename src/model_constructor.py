import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks
import os

ACTIVATION_FUNCTION: str = "relu"
N_HIDDEN: int = 4


def system_setup():

    n_cpu = os.cpu_count()
    print(f"Number of CPU cores:\t\t{n_cpu}")

    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    print(f"Number of GPUs available:\t{len(physical_devices)}")

    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)


def construct_fft_net_model(
    n_hidden_layers: int, training_samples_dict: dict, l2: float = 1e-3, dropout: float = 0.2, negative_slope: float = 0.3
) -> keras.Sequential:
    
    input_shape = training_samples_dict["samples"].shape[1:]
    output_shape = training_samples_dict["labels"].shape[1]

    model = keras.Sequential()
    model.add(layers.Dense(2048, input_shape=input_shape))

    # add the number of hidden layers
    for _ in range(n_hidden_layers):
        model.add(layers.Dense(units = 1024, kernel_regularizer=regularizers.L2(l2)))
        model.add(layers.LeakyReLU(negative_slope=negative_slope))
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(output_shape, activation="sigmoid"))

    model.summary()
    return model


def compile_model(model: keras.Sequential, learning_rate: float, momentum: float, threshold: float):
    model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
    loss = keras.losses.CategoricalCrossentropy(), # for non-binary classification
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision", thresholds=threshold),
        keras.metrics.Recall(name="recall", thresholds=threshold),
    ]
    )

def train_model(model: keras.Sequential, samples_dict, epochs: int, batch_size: int, validation_split: float = 0.1, use_early_stopping: bool = True):
    callback_list = []
    if use_early_stopping:
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        callback_list.append(early_stopping)
    
    return model.fit(x=samples_dict["samples"],
    y=samples_dict["labels"],
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    shuffle=True,
    callbacks=callback_list, 
    class_weight=samples_dict["class_weights"])


def evaluate(model: keras.Sequential, test_samples_dict: dict):
    return model.evaluate(test_samples_dict["samples"], test_samples_dict["labels"], verbose = 1, return_dict=True)

def predict(model: keras.Sequential, test_samples_dict:dict):
    prediction = model.predict(test_samples_dict["samples"])
    true_labels = test_samples_dict["labels"].argmax(axis=1)
    predicted_labels = prediction.argmax(axis=1)

    true_labels = test_samples_dict["encoder"].inverse_transform(true_labels)
    predicted_labels = test_samples_dict["encoder"].inverse_transform(predicted_labels)

    print("\nMODEL PREDICTIONS\n")
    print(true_labels)
    print(predicted_labels)

    return true_labels, predicted_labels