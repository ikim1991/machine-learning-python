# Decision Tree & Random Forest Regression

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
dataset = pd.read_csv('./dataset/Position_Salaries.csv')
X = dataset.iloc[:,1].values
y = dataset.iloc[:,-1].values

# Fit Decision Tree Regression Model from scikit learn
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor(random_state = 0)
decision_tree.fit(X.reshape(-1,1), y)

# Fit Random Forest Regression Model from scikit Learn
from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor(n_estimators = 100, random_state = 0)
random_forest.fit(X.reshape(-1,1), y)

# Predicting a new result
y_dt = decision_tree.predict([[6.5]])
y_rf = random_forest.predict([[6.5]])

# Visualizing the Decision Tree Model
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, decision_tree.predict(X_grid), 'r')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Salary by Position Level (Decision Tree)')
plt.show()

# Visualizing the Decision Tree Model
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y)
plt.plot(X_grid, random_forest.predict(X_grid), 'r')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Salary by Position Level (Random Forest n=300)')
plt.show()
