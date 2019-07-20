# Linear Regression

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
dataset = pd.read_csv('./dataset/Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Split Dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Fit and Predict Linear Regression Model from scikit learn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Visualizing the Training Set
plt.scatter(X_train, y_train, color='r')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.xlabel('Year Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Experience (Training Set)')
plt.show()

# Visualizing the Test Set
plt.scatter(X_test, y_test, color='r')
plt.plot(X_train, regressor.predict(X_train), color='b')
plt.xlabel('Year Experience')
plt.ylabel('Salary')
plt.title('Salary vs. Experience (Test Set)')
plt.show()
