from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.io as sio

# Load the digits dataset
digits = datasets.load_digits()
X = digits.data      # shape: (n_samples, 64)
y = digits.target    # shape: (n_samples,)

# Scale the data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# One-hot encode the labels.
one_hot_y = np.eye(10)[y]  # shape: (n_samples, 10)

# Sort the data by class labels (from 0 to 9)
sorted_indices = np.argsort(y)
X_scaled_sorted = X_scaled[sorted_indices]
one_hot_y_sorted = one_hot_y[sorted_indices]

# Append the one-hot encoded labels to the scaled features.
data = np.hstack([X_scaled_sorted, one_hot_y_sorted])  # Final shape: (n_samples, 64+10=74)

# Save the resulting data to a .mat file with the variable name 'data'
sio.savemat('digits.mat', {'data': data})
