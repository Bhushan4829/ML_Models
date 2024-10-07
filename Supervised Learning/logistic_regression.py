import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load data
data = pd.read_csv(r"D:\Downloads\ML_Models\ML_Models\Dataset\FuelConsumption.csv")

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

class LogisticRegression():
    def __init__(self,lr=0.01,epochs = 10):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    def sigmoid_function(z):
        if z>=0:
            y = (1/1+np.exp(z))
        else:
            y = (np.exp(z)/1 + np.exp(z))
        return y
    def _sigmoid(self,X):
        return np.array([self.sigmoid_function(value) for value in X])
    def prediction(self,X):
        
    def fit(self,X,y):
        self.weights = np.random.rand(X.shape[0])
        self.bias = 0
        self.X = X
        self.y = y
        for i in range(self.epochs):
            x_dot_weights = np.matmul(self.weights,X.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            y_zero_loss = y*np.log(pred)
            y_one_loss = (1-y)*np.log(pred)
            loss = -np.mean(y_zero_loss+y_one_loss)
            db = np.mean(pred-y)
            gradient_w = np.matmul(self.X.T,pred-y)
            dw = np.array([np.mean(grad) for grad in gradient_w])
            self.weights = self.weights - 0.1*dw
            self.bias = self.bias - 0.1*db
            





#References
#https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/