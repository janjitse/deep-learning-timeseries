from pathlib import Path

# from re import S
from typing import Tuple

import numpy as np

from tensorflow import keras


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    filename = Path("jena_climate_2009_2016.csv")

    with open(filename) as f:
        data = f.read()

    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:]

    temperature = np.zeros((len(lines),))
    raw_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(",")[1:]]
        temperature[i] = values[1]
        raw_data[i, :] = values[:]

    return temperature, raw_data


def split_data(raw_data, temperature):
    num_train_samples = int(0.5 * len(raw_data))
    num_val_samples = int(0.25 * len(raw_data))
    num_test_samples = len(raw_data) - num_train_samples - num_val_samples

    train_temperature, train_raw_data = (
        temperature[:num_train_samples],
        raw_data[:num_train_samples],
    )

    val_temperature, val_raw_data = (
        temperature[num_train_samples : num_train_samples + num_val_samples],
        raw_data[num_train_samples : num_train_samples + num_val_samples],
    )

    test_temperature, test_raw_data = (
        temperature[num_train_samples + num_val_samples :],
        raw_data[num_train_samples + num_val_samples :],
    )

    mean = train_raw_data.mean(axis=0)
    train_raw_data -= mean
    val_raw_data -= mean
    test_raw_data -= mean

    std = train_raw_data.std(axis=0)
    train_raw_data /= std
    val_raw_data /= std
    test_raw_data /= std

    return (
        train_raw_data,
        train_temperature,
        val_raw_data,
        val_temperature,
        test_raw_data,
        test_temperature,
    )


def normalize_data(raw_data, num_train_samples):
    train_raw_data = raw_data[:num_train_samples, :]
    mean = train_raw_data.mean(axis=0)
    train_raw_data = train_raw_data - mean
    raw_data = raw_data - mean

    std = train_raw_data.std(axis=0)
    raw_data = raw_data / std
    return raw_data, mean, std


def split_data_keras(
    raw_data, temperature, sampling_rate=6, sequence_length=120, batch_size=256
):

    num_train_samples = int(0.5 * len(raw_data))
    num_val_samples = int(0.25 * len(raw_data))
    delay = sampling_rate * (sequence_length + 24 - 1)
    train_dataset = keras.utils.timeseries_dataset_from_array(
        raw_data[:-delay],
        targets=temperature[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=True,
        batch_size=batch_size,
        start_index=0,
        end_index=num_train_samples,
    )

    val_dataset = keras.utils.timeseries_dataset_from_array(
        raw_data[:-delay],
        targets=temperature[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=num_train_samples,
        end_index=num_train_samples + num_val_samples,
    )

    test_dataset = keras.utils.timeseries_dataset_from_array(
        raw_data[:-delay],
        targets=temperature[delay:],
        sampling_rate=sampling_rate,
        sequence_length=sequence_length,
        shuffle=False,
        batch_size=batch_size,
        start_index=num_train_samples + num_val_samples,
    )
    return train_dataset, val_dataset, test_dataset


def evaluate_naive_method(dataset, mean=0, std=1):
    total_abs_err = 0
    samples_seen = 0
    for samples, targets in dataset:
        # Temperature is at column 1 of the dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen
