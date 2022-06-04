import os
from pathlib import Path

import numpy as np

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

# import matplotlib.pyplot as plt

# fig = plt.plot(range(len(temperature)), temperature)
# plt.savefig("test.png")
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

mean = train_raw_data.mean(axis=0)
raw_data -= mean

std = train_raw_data.std(axis=0)
raw_data /= std


from tensorflow import keras

SAMPLING_RATE = 6
SEQUENCE_LENGTH = 120
DELAY = SAMPLING_RATE * (SEQUENCE_LENGTH + 24 - 1)
BATCH_SIZE = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-DELAY],
    targets=temperature[DELAY:],
    sampling_rate=SAMPLING_RATE,
    sequence_length=SEQUENCE_LENGTH,
    shuffle=True,
    batch_size=BATCH_SIZE,
    start_index=0,
    end_index=num_train_samples,
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-DELAY],
    targets=temperature[DELAY:],
    sampling_rate=SAMPLING_RATE,
    sequence_length=SEQUENCE_LENGTH,
    shuffle=True,
    batch_size=BATCH_SIZE,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples,
)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-DELAY],
    targets=temperature[DELAY:],
    sampling_rate=SAMPLING_RATE,
    sequence_length=SEQUENCE_LENGTH,
    shuffle=True,
    batch_size=BATCH_SIZE,
    start_index=num_train_samples + num_val_samples,
)


def evaluate_naive_method(dataset):
    total_abs_err = 0
    samples_seen = 0
    for samples, targets in dataset:
        # Temperature is at column 1 of the dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen


print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}.")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}.")


from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(SEQUENCE_LENGTH, raw_data.shape[-1]))
x = layers.SeparableConv1D(16, 24, activation="relu", padding="same")(inputs)
x = layers.MaxPooling1D(2)(x)
x = layers.SeparableConv1D(32, 12, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(2)(x)
x = layers.SeparableConv1D(64, 6, activation="relu", padding="same")(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)]

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
print(model.summary())

history = model.fit(
    train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks
)
model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
