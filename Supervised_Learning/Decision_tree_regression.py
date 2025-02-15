#What's Left? Understanding of the code how it is working and how to improve.
#Moreover how we impkemented Decision Tree from scratch and from library.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from collections import Counter
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
# scaledtarget = featurescaling.fit_transform(y)
# scaledtgt = pd.DataFrame(scaledtarget, columns=["credit_card_default"])
scaled_df = pd.DataFrame(scaleddata, columns=X.columns)
print(scaled_df.head())
# Train Test Split

train_x, test_x, train_y, test_y = train_test_split(scaled_df, y, test_size=0.2, random_state=42)

# Decision Tree Implementation from library
model = DecisionTreeRegressor()
model.fit(train_x, train_y)
print(model.score(test_x, test_y))
print(model.predict(test_x))
# Decision Tree Implementation from scratch
#TreeNode
class TreeNode:
    def __init__(self,data, feature_index, feature_value, prediction_probs, information_gain) -> None:
        self.data = data #Stores subset of training data that reaches this particular node(Contains both features and labels)
        self.feature_index = feature_index #Indicates the feature index of feature that is used for splitting
        self.feature_value = feature_value # Stores threshold value of feature that is used for splitting
        self.prediction_probs = prediction_probs#Represents probability distribution of labels in the node
        self.information_gain = information_gain #Represents information gain of the split
        self.feature_importance = self.data.shape[0] * self.information_gain#Stores importance of feature used for splitting
        self.left = None#left child of the node
        self.right = None#right child of the node

    def node_info(self):
        print("Feature Index: ", self.feature_index)
        print("Feature Value: ", self.feature_value)
        print("Information Gain: ", self.information_gain)
        print("Prediction Probs: ", self.prediction_probs)
        print("Feature Importance: ", self.feature_importance)
        print("Data Shape: ", self.data.shape)

class DecisionTreeScratch():
    """
    Decision Tree Regression
    Training: Use "train" function with train set features and labels
    Prediction: Use "predict" function with test set features
    """
    def __init__(self,max_depth=4,min_samples_leaf=1,min_information_gain=0.0, num_of_features_split = None, amount_of_say=None)->None:
        # Class with Hyperparameters
        self.max_depth = max_depth #Maximum depth of the tree
        self.min_samples_leaf = min_samples_leaf#Minimum number of samples required to be at a leaf node
        self.min_information_gain = min_information_gain#Minimum information gain required for splitting
        self.num_of_features_split = num_of_features_split#Number of features to consider for best split
        self.amount_of_say = amount_of_say#Amount of say of each feature in the tree(Used for adaboost)
        self.tree = None#Root Node of the tree
    def _entropy(self,class_probs:list)->float:
        """ 
        Calculate Entropy of a node
        What is Entropy? A measure of uncertainty or impurity within a node of data
        Indicates how uncertain prediction would be if a split were made at that point
        """
        return -sum([p * np.log2(p) for p in class_probs if p > 0])
    def _class_probabilities(self,labels:list)->list:
        """
        Calculate Class Probabilities
        Can use np.bincount as well in place of Counter, but major difference is that Counter returns a dictionary, while np.bincount returns a numpy array, moreover np.bincount assumes range from 0 to max value in the array.
        """
        total_count = len(labels)
        return [count / total_count for count in Counter(labels).values()]
    def _data_entropy(self,labels:list)->float:
        class_probs = self._class_probabilities(labels)
        return self._entropy(class_probs)
    def _partition_entropy(self,subsets:list)->float:
        """
        subsets: List of subsets of data (EX: [[1,0,0],[1,1,1]])
        """
        total_count = sum(len(subset) for subset in subsets)
        return sum(self._data_entropy(subset) * len(subset) / total_count for subset in subsets)
    def _split(self,data:np.array,feature_idx:int,feature_val:float)->list:
        """
        Split data based on threshold value of feature
        """
        left_indices = np.where(data[:,feature_idx] <= feature_val)
        right_indices = np.where(data[:, feature_idx] > feature_val)[0]
        return data[left_indices], data[right_indices]
    def _select_features_to_use(self,data:np.array) ->list:
        """
        Select a subset of features to consider for splitting
        """
        feature_idx = list(range(data.shape[1] - 1))
        if self.num_of_features_split == "sqrt":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))), replace=False)
        elif self.num_of_features_split == "log":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))), replace=False)
        else:
            feature_idx_to_use = feature_idx
        return feature_idx_to_use
    def _find_best_split(self,data:np.array)->tuple:
        """
        Find the best split for the data (with the lowest entropy) given data
        """
        #Get all unique values of features
        min_part_entropy = float("inf")
        best_feature_idx = None
        best_feature_val = None     
        left_min, right_min = None, None
        feature_idx_to_use = self._select_features_to_use(data)
        for idx in feature_idx_to_use:
            feature_vals = np.percentile(data[:, idx], q=[25, 50, 75])
            for feature_val in feature_vals:
                left, right = self._split(data, idx, feature_val)
                if len(left)<self.min_samples_leaf or len(right)<self.min_samples_leaf:
                    continue
                partition_entropy = self._partition_entropy([left[:,-1], right[:,-1]])
                if partition_entropy < min_part_entropy:
                    min_part_entropy = partition_entropy
                    best_feature_idx = idx
                    best_feature_val = feature_val
                    left_min,right_min = left, right
        return left_min, right_min,best_feature_idx, best_feature_val, min_part_entropy
    def _find_label_probs(self,data:np.array)->list:
        """
        Find the probability distribution of labels in the data
        """
        labels_as_integers = data[:,-1].astype(int)
        total_labels = len(labels_as_integers)
        label_probs = np.zeros(len(self.labels_in_training_data),dtype=float)
        for i,label in enumerate(self.labels_in_training_data):
            label_inx = np.where(labels_as_integers == i)[0]
            if len(label_inx) > 0:
                label_probs[i] = len(label_inx) / total_labels
        return label_probs
    def _create_tree(self,data:np.array,current_depth:int) -> TreeNode:
        """
        Recursively create the tree
        """

        if current_depth > self.max_depth or len(data) < self.min_samples_leaf:
            return None
        split_left, split_right, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)
        label_probs = self._find_label_probs(data)
        node_entropy = self._entropy(label_probs)
        information_gain = node_entropy - split_entropy
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probs, information_gain)
        if self.min_samples_leaf>split_right.shape[0] or self.min_samples_leaf>split_left.shape[0]:
            return node
        elif information_gain < self.min_information_gain:
            return node
        current_depth += 1
        node.left = self._create_tree(split_left, current_depth)
        node.right = self._create_tree(split_right, current_depth)
        return node
    def _predict_one_sample(self,X:np.array) -> np.array:
        """ Returns Prediction for one sample"""
        node = self.tree
        while node:
            if node.feature_index is None:
                return node.prediction_probs
            if X[node.feature_index] <= node.feature_value:
                if node.left:
                    node = node.left
                else:
                    return node.prediction_probs
            else:
                if node.right:
                    node = node.right
                else:
                    return node.prediction_probs
        return node.prediction_probs

    def train(self,X:np.array,y:np.array)->None:
        """
        Trains the model with given X and Y datasets
        """
        self.labels_in_training_data = np.unique(y)
        train_data = np.concatenate((X,y.reshape(-1,1)),axis=1)
        self.tree = self._create_tree(train_data,0)
        # self.feature_importances = dict.fromkeys(range(X.shape[1]),0)
        # self._calculate_feature_importances(self.tree)
        # self.feature_importances = {k: v for k, v / total for total in (sum(self.feature_importances.values()),) for k, v in self.feature_importances.items()}
    def predict_probs(self,X:np.array)->np.array:
        """
        Returns Predictions for given X dataset
        """
        return np.apply_along_axis(self._predict_one_sample,1,X)
    def predict(self,X:np.array)->np.array:
        """
        Returns Predictions for given X dataset
        """
        pred_probs = self.predict_probs(X)
        return np.argmax(pred_probs,axis=1)
    def _calculate_feature_importances(self,node):
        if node!=None:
            self.feature_importances[node.feature_index] += node.feature_importance
            self._calculate_feature_importances(node.left)
            self._calculate_feature_importances(node.right)
#Training Scratch Model
scratch_model = DecisionTreeScratch(max_depth=4,min_samples_leaf=1,min_information_gain=0.0)
scratch_model.train(train_x.values,train_y.values)
#Probabilities
print(scratch_model.predict_probs(test_x.values))
#Evaluation
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss

# Get predicted probabilities
predicted_probs = scratch_model.predict_probs(test_x.values)

# Clip probabilities to prevent log errors
predicted_probs = np.clip(predicted_probs, 1e-15, 1 - 1e-15)

# Convert to binary predictions using 0.5 threshold
predicted_classes = (predicted_probs[:, 1] >= 0.5).astype(int)

# Compute Metrics
logloss = log_loss(test_y, predicted_probs[:, 1])  # Use only class 1 probabilities
roc_auc = roc_auc_score(test_y, predicted_probs[:, 1])  # Use only class 1 probabilities
brier_score = brier_score_loss(test_y, predicted_probs[:, 1])  # Use only class 1 probabilities
accuracy = accuracy_score(test_y, predicted_classes)
precision = precision_score(test_y, predicted_classes)
recall = recall_score(test_y, predicted_classes)
f1 = f1_score(test_y, predicted_classes)

# Print Results
print("Log Loss:", logloss)
print("ROC-AUC Score:", roc_auc)
print("Brier Score:", brier_score)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


#Testing on Test data

# Your understanding of the data , what you learnt and also all possible options you can explore.


#References:
#https://medium.com/@beratyildirim/regression-tree-from-scratch-using-python-a74dba2bba5f
#https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
#Code: https://github.com/enesozeren/machine_learning_from_scratch/tree/main/decision_trees