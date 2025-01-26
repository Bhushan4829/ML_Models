#Formatting and Update required with comments
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load data
data = pd.read_csv(r"Dataset\age_predictions_cleaned.csv")

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

# # Ensure binary column is selected for target
features_train = train.drop(columns=['age_group'], axis=1)
target_train = train['age_group']

features_test = test.drop(columns=['age_group'], axis=1)
target_test = test['age_group']

# #Feature Scaling
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

class LogisticRegression():
    def __init__(self,lr=0.01,epochs = 10):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    def fit(self,X,y):
        self.weights = np.zeros(X.shape[1])  # Initialize weights with zeros
        self.bias = 0
        self.X = X
        self.y = y
        for _ in range(self.epochs):
            x_dot_weights = np.dot(X,self.weights) + self.bias
            epsilon = 1e-15
            pred = self._sigmoid(x_dot_weights)  # Ensure predictions are in range (epsilon, 1 - epsilon)
            y_zero_loss = y * np.log(pred+epsilon)
            y_one_loss = (1 - y) * np.log(1 - pred+epsilon)
            loss = -np.mean(y_zero_loss + y_one_loss)

            dw = np.dot(X.T, (pred - y)) / len(y)
            db = np.mean(pred - y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
            if _%10==0:
                print(f"Epoch:{_} Loss:{loss}")
        return self.weights,self.bias
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_output)
        return (y_pred >= 0.5).astype(int) 

    def accuracy(self, y_pred, y_true):
        return round(np.sum(y_pred == y_true) / len(y_true) * 100, 2)
    

Lr = LogisticRegression(lr=0.01, epochs=100)
model = Lr.fit(features_train_normalized.values, target_train.values)

y_pred_train = Lr.predict(features_train_normalized.values)
y_pred_test = Lr.predict(features_test_normalized.values)

y_pred_train = Lr.predict(features_train_normalized.values)
y_pred_test = Lr.predict(features_test_normalized.values)

train_accuracy = Lr.accuracy(y_pred_train,target_train.values)
test_accuracy = Lr.accuracy(y_pred_test,target_test.values)
print(f"Training Accuracy: {train_accuracy}%")
print(f"Testing Accuracy: {test_accuracy}%")





#References
#https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/