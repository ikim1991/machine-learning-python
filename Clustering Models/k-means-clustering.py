# Upper Confidence Bound

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
dataset = pd.read_csv('./dataset/Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
