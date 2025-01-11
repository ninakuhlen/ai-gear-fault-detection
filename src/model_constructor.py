import tensorflow as tf
from tensorflow import keras
from keras import layers
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
    number_of_hidden_layers: int, input_shape: tuple
) -> keras.Sequential:
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(2048))

    # add the number of hidden layers
    for _ in range(number_of_hidden_layers):
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(negative_slope=0.3))

    model.add(layers.Dense(5, activation="sigmoid"))
    model

    model.summary()
    return model
