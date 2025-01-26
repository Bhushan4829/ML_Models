import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# training_data = pd.read_csv("D:\\Downloads\\ML_Models\\american_express_data\\train.csv")
training_data = pd.read_csv(r"Dataset\american_express_data\train.csv")
# print(training_data["gender"].unique().sum())

# Drop Name and Customer Id column as they do not add any predicive value and those columns are unique identifiers.
training_data = training_data.drop(["name", "customer_id"], axis=1)
# print(training_data.head())
# One hot Encode on Gender, Occupaton_type, Owns_car, Owns_house, also check for any unknown, none, null values in this columns.
missing_values = training_data[["gender", "owns_car", "owns_house", "occupation_type"]].isnull().sum()
# print("Missing Values:\n", missing_values)
training_data["gender"] = training_data["gender"].fillna(-1)
training_data["occupation_type"] = training_data["occupation_type"].fillna("Unknown")
training_data["gender"] = training_data["gender"].map({"F": 0, "M": 1})
training_data["owns_car"] = training_data["owns_car"].map({"N": 0, "Y": 1})
training_data["owns_house"] = training_data["owns_house"].map({"N": 0, "Y": 1})
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_occupation = encoder.fit_transform(training_data[["occupation_type"]])
encoded_df = pd.DataFrame(encoded_occupation, columns=encoder.get_feature_names_out(["occupation_type"]))
training_data = pd.concat([training_data, encoded_df], axis=1).drop("occupation_type", axis=1)

# Feature Scaling
X = training_data.drop("credit_card_default", axis=1)
y = training_data["credit_card_default"]
featurescaling = StandardScaler()
scaleddata = featurescaling.fit_transform(X)
scaled_df = pd.DataFrame(scaleddata, columns=X.columns)
print(scaled_df.head())
# Train Test Split

train_x, test_x, train_y, test_y = train_test_split(scaled_df, y, test_size=0.2, random_state=42)

# Decision Tree Implementation from library
model = DecisionTreeClassifier()
model.fit(train_x, train_y)
print(model.score(test_x, test_y))
print(model.predict(test_x))
# Decision Tree Implementation from scratch

# Predictions

#Testing on Test data

# Your understanding of the data , what you learnt and also all possible options you can explore.


#References:
#https://medium.com/@beratyildirim/regression-tree-from-scratch-using-python-a74dba2bba5f
#https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
#Code: https://github.com/enesozeren/machine_learning_from_scratch/tree/main/decision_trees