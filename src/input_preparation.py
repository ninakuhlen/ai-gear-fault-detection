import numpy as np
from pandas import DataFrame, concat
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def compare_merge_attributes(attributes_a: dict, attributes_b: dict):
    # just copy attributes_b, if an empty dict attributes_a is given
    if not attributes_a:
        return attributes_b
    
    assert set(attributes_a.keys()) == set(attributes_b.keys()), "Old and new attribute dictionaries have incompatible keys!"

    keys = attributes_b.keys()
    merged_attributes = {}

    for key in keys:
        value_a = attributes_a[key]
        value_b = attributes_b[key]

        if value_a == value_b:
            # if values are the same, use one of them
            merged_attributes[key] = value_a
        else:
            # if values are different, create or append to a list
            if isinstance(value_a, list):
                merged_attributes[key] = value_a + ([value_b] if value_b not in value_a else [])
            elif isinstance(value_b, list):
                merged_attributes[key] = ([value_a] if value_a not in value_b else []) + value_b
            else:
                merged_attributes[key] = [value_a, value_b]

    return merged_attributes


def concatenate_datasets(datasets: list[DataFrame], use_binary_labeling: bool = False):
    keys = []
    attrs = {}
    for index, dataset in enumerate(datasets):
        keys.append(dataset.attrs["path"].stem)

        attrs = compare_merge_attributes(attrs, dataset.attrs)

        if use_binary_labeling:
            if dataset.attrs["unbalance"] == "none":
                dataset["label"] = "none"
            else:
                dataset["label"] = "some"

        dataset["label"] = dataset.attrs["unbalance"]
        datasets[index] = dataset
    
    concatenated_datasets = concat(datasets, keys=keys)
    concatenated_datasets.attrs = attrs

    return concatenated_datasets


def split_data(dataframe: DataFrame, data_columns: list[str], random_state: int = 0):

    sample_rate = dataframe.attrs["sample_rate"]

    samples = dataframe[data_columns].to_numpy()
    samples = np.reshape(samples, (-1, sample_rate, len(data_columns)))

    # TODO check position of len(data_columns)

    encoder = LabelEncoder()
    
    labels = dataframe["label"].to_numpy()
    n_classes = len(np.unique(labels))
    labels = np.reshape(labels, (-1, sample_rate))[:,0]
    encoded_labels = encoder.fit_transform(labels)
    one_hot_encoded_labels = to_categorical(encoded_labels, n_classes)

    # generate initial class weights
    class_weights = {}
    for index in range(n_classes):
        class_weights[index] = 1.0

    # shuffle samples and pack in dictionary
    shuffled_samples, shuffled_labels = shuffle(samples, one_hot_encoded_labels, random_state=random_state)

    sample_dict = {"samples": shuffled_samples.squeeze(),
            "labels": shuffled_labels,
            "encoder": encoder,
            "class_weights": class_weights}
    
    return sample_dict


def check_data(sample_dict: dict):
    one_hot_encoded_labels = sample_dict["labels"]
    encoded_labels = np.argmax(one_hot_encoded_labels, axis=1)
    labels = sample_dict["encoder"].inverse_transform(encoded_labels)

    unique_labels = np.unique(labels)
    class_counts = np.sum(one_hot_encoded_labels, axis=0)

    print("\nCHECK SAMPLE DISTRIBUTION\n")

    for entry in zip(unique_labels, class_counts):
        print(f"\tClass '{entry[0]}':\t{entry[1]} samples")



