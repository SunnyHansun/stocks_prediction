# stocks_prediction

# Data Processing Notes
The data is normalized using sklearn StandardScaler and organized into 20-day sliding window training samples. These can all be converted to tensors if we want to use pyTorch. Here's the resulting files:

Data.npz: load using numpy, contains 3D array

    Features: 
    [curr open / previous close, 
    curr close / previous close, 
    (high-low) / previous close]

    Target:
    next day's close / curr day's close

    Shape:
    X_train, y_train: (15173, 20, 3), (15173, 1)
    X_val, y_val: (5045, 20, 3), (5045, 1)
    X_test, y_test: (5045, 20, 3), (5045, 1)

    e.g. X_train is all training windows        (15173, 20, 3)
    X_train[0] is the first 20 day window       (20, 3)
    X_train[0][0] is the features of the first day within the first window (3, )

feature_scaler.pkl and target_scaler.pkl: load using joblib (part of sklearn)

    Both files constains sklearn StandardScaler objects used to normalize the input and target values. Can be used to reverse the normalization using inverse_transform().

may need to "python3 -m pip install pandas numpy scikit-learn" if you dont have these libraries