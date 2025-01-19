import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load data
data = pd.read_csv(".\Dataset\FuelConsumption.csv")

# Dropping unnecessary columns
data = data.drop(columns=['Year', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'TRANSMISSION', 'FUEL'], axis=1)

# Drop rows with any missing values
data = data.dropna()

# Shuffle data
data = data.sample(frac=1)

# Split the data into training and testing sets
ratio = 0.75
total_rows = data.shape[0]
train_size = int(total_rows * ratio)

train = data[:train_size]
test = data[train_size:]

# Ensure 'COEMISSIONS' is used correctly without trailing space
features_train = train.drop(columns=['COEMISSIONS '], axis=1)
target_train = train['COEMISSIONS ']

features_test = test.drop(columns=['COEMISSIONS '], axis=1)
target_test = test['COEMISSIONS ']

#Feature Scaling
# scaler = StandardScaler()
# features_train = scaler.fit_transform(features_train)
# features_test = scaler.transform(features_test)

means = features_train.mean(axis=0)
stds = features_train.std(axis=0)
features_train_normalized = (features_train - means) / stds
features_test_normalized = (features_test - means) / stds


# Check dimensions
print(f"features_train shape: {features_train.shape}")
print(f"target_train shape: {target_train.shape}")
print(f"features_test shape: {features_test.shape}")
print(f"target_test shape: {target_test.shape}")

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.random.rand(num_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/num_samples) * np.dot(X.T, y_pred - y)
            db = (1/num_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def cost_function(Y, Y_pred):
    return (1/len(Y)) * np.sum((Y_pred - Y) ** 2)

# Instantiate and train the model
LR = LinearRegression()
model = LR.fit(features_train_normalized, target_train)

# Predict and calculate the cost
pred = model.predict(features_test_normalized)
print("Cost Function:", cost_function(target_test, pred))


#References
# https://www.kdnuggets.com/linear-regression-from-scratch-with-numpy
# https://www.geeksforgeeks.org/how-to-split-data-into-training-and-testing-in-python-without-sklearn/
# https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-normalize-or-standardize-a-dataset-in-python.md
# Dataset
# https://www.kaggle.com/datasets/krupadharamshi/fuelconsumption
