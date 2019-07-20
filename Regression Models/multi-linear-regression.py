# Multi-Linear Regression

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
dataset = pd.read_csv('./dataset/50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Multicolinearity of our categorical data
X = X[:,1:]

# Split Dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Fit and Predict Linear Regression Model from scikit learn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Model Optimization using Backward Elimination
import statsmodels.formula.api as sm
# Add a column of ones to account for our constant or intercept value
X = np.append(np.ones((len(X), 1)), X, axis = 1)
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog = X_opt).fit()
regressor_OLS.summary()
