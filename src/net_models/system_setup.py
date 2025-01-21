import os
import tensorflow as tf


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
