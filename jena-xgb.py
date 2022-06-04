from pathlib import Path
import numpy as np
import xgboost as xgb

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

num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples

SAMPLING_RATE = 6
SEQUENCE_LENGTH = 120
DELAY = SAMPLING_RATE * (SEQUENCE_LENGTH + 24 - 1)

targets = temperature[DELAY:]

train_temperature, train_raw_data = (
    targets[:num_train_samples],
    raw_data[:num_train_samples],
)

val_temperature, val_raw_data = (
    targets[num_train_samples : num_train_samples + num_val_samples],
    raw_data[num_train_samples : num_train_samples + num_val_samples],
)

print("Processing training data")
processed_train_data = np.zeros(
    (len(train_raw_data) - SEQUENCE_LENGTH, SEQUENCE_LENGTH * train_raw_data.shape[1])
)

for idx in range(len(processed_train_data)):
    processed_train_data[idx, :] = train_raw_data[
        idx : idx + SEQUENCE_LENGTH, :
    ].ravel()

print("Processing validation data")
processed_val_data = np.zeros(
    (len(val_raw_data) - SEQUENCE_LENGTH, SEQUENCE_LENGTH * val_raw_data.shape[1])
)
for idx in range(len(processed_val_data)):
    processed_val_data[idx, :] = val_raw_data[idx : idx + SEQUENCE_LENGTH, :].ravel()


model = xgb.XGBRegressor()

print("Fitting model")

model.fit(X=processed_train_data, y=train_temperature[SEQUENCE_LENGTH:])
print("Evaluating model")
y_val = model.predict(processed_val_data)
print(np.mean(np.abs(y_val - val_temperature[SEQUENCE_LENGTH:])))

# model.
