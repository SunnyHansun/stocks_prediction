import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# may need to "python3 -m pip install pandas numpy scikit-learn" if you dont have these libraries

# ==== Load and clean data ====
# load data from sp500_oc29.csv
data = pd.read_csv("sp500_oct29.csv", skiprows=2, skipfooter=2, engine="python")
data.columns = data.columns.str.lower()

# convert dates column to datetime and prices to numeric values
data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="coerce")

for col in ["open", "high", "low", "close"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# only keep data from 1930 onwards (for some reason they don't have 1930-01-01)
data = data[data["date"] >= "1930-01-02"].copy()
data = data.sort_values("date").reset_index(drop=True)

# previous day's closing index
previous_close = data["close"].shift(1)

# construct features relative to the previous close
data["open_relative"] = data["open"] / previous_close
data["close_relative"] = data["close"] / previous_close
data["range_relative"] = ((data["high"] - data["low"]) / previous_close)

feature_cols = ["open_relative", "close_relative", "range_relative"]

# create the targets as next day's relative closing index
data["target"] = data["close"].shift(-1) / data["close"]

# remove rows with missing or invalid feature/target values
cols = feature_cols + ["target"]
valid_rows = np.isfinite(data[cols]).all(axis=1)
data = data[valid_rows].reset_index(drop=True)


# ==== Splitting data ====
# calculate split indices
n = len(data)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

# split chronologically (60/20/20) to ensure no data leakage between the sets
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]


# ==== Normalizing the features ====
feature_scaler = StandardScaler()
feature_scaler.fit(train_data[feature_cols]) # calculate mean & std dev using training set

# normalize all three data sets using training set's stats
# formula: x_scaled = (x - mew_train) / std_dev_train
X_train = feature_scaler.transform(train_data[feature_cols])
X_val = feature_scaler.transform(val_data[feature_cols])
X_test = feature_scaler.transform(test_data[feature_cols])

# debugging prints
"""print("Training data:")
print(X_train[:5])

print("Validation data:")
print(X_val[:5])

print("Testing data:")
print(X_test[:5])"""

# normalize target
target_scaler = StandardScaler()
target_scaler.fit(train_data[["target"]])

y_train = target_scaler.transform(train_data[["target"]])
y_val = target_scaler.transform(val_data[["target"]])
y_test = target_scaler.transform(test_data[["target"]])

# debugging prints
#print("Training targets:")
#print(y_train[:5])

# ==== Create sliding window ==== 
def create_window(features, targets, size=20):
    """
    Returns the new inputs and targets after applying sliding window
    """
    X, y = [], []
    num_rows = len(features)

    # for every window, append the corresponding data to input and target
    for i in range(num_rows - size + 1):
        X.append(features[i: i + size])
        y.append(targets[i + size - 1]) # targets previously shifted by -1
    
    return np.array(X), np.array(y)

# apply sliding window
X_train, y_train = create_window(X_train, y_train)
X_val, y_val = create_window(X_val, y_val)
X_test, y_test = create_window(X_test, y_test)

# save the datasets in .npz (csv need to flatten 3D array into 2D)
np.savez_compressed(
    "data.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

# save the training and target scalers for reversing 
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")
