import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

training_data = pd.read_csv("Dataset\\american_express_data\\train.csv")
print(training_data.describe())