import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# training_data = pd.read_csv("D:\\Downloads\\ML_Models\\american_express_data\\train.csv")
training_data = pd.read_csv(r"Dataset\american_express_data\train.csv")
print(training_data["gender"].unique().sum())

# Drop Name and Customer Id column as they do not add any predicive value and those columns are unique identifiers.

# One hot Encode on Gender, Occupaton_type, Owns_car, Owns_house, also check for any unknown, none, null values in this columns.

# Feature Scaling

# Train Test Split

# Decision Tree Implementation from scratch

# Predictions

#Testing on Test data

# Your understanding of the data , what you learnt and also all possible options you can explore.


#References:
#https://medium.com/@beratyildirim/regression-tree-from-scratch-using-python-a74dba2bba5f
#https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
#Code: https://github.com/enesozeren/machine_learning_from_scratch/tree/main/decision_trees