# Polynomial Regression

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
dataset = pd.read_csv('./dataset/Position_Salaries.csv')
X = dataset.iloc[:,1].values
y = dataset.iloc[:,-1].values

# Fit Linear Regression Model from scikit learn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X.reshape(-1,1), y)

# Fit Polynomial Features to our Linear Regression Model
from sklearn.preprocessing import PolynomialFeatures
poly_four = PolynomialFeatures(degree=4)
X_poly4 = poly_four.fit_transform(X.reshape(-1,1))
poly_three = PolynomialFeatures(degree=3)
X_poly3 = poly_three.fit_transform(X.reshape(-1,1))
poly_two = PolynomialFeatures(degree=2)
X_poly2 = poly_two.fit_transform(X.reshape(-1,1))
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly2, y)
lin_reg3 = LinearRegression()
lin_reg3.fit(X_poly3, y)
lin_reg4 = LinearRegression()
lin_reg4.fit(X_poly4, y)

# Visualizing the differnt Degrees of the Polynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='r')
plt.plot(X_grid, lin_reg.predict(X_grid), color='b', label='Degree: 1')
plt.plot(X_grid, lin_reg2.predict(poly_two.fit_transform(X_grid.reshape(-1,1))), label='Degree: 2')
plt.plot(X_grid, lin_reg3.predict(poly_three.fit_transform(X_grid.reshape(-1,1))), label='Degree: 3')
plt.plot(X_grid, lin_reg4.predict(poly_four.fit_transform(X_grid.reshape(-1,1))), label='Degree: 4')
plt.legend()
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Salary by Position Level (Polynomial Regression)')
plt.show()

# Predictions of Salary by Degree of Polynomial Features for Level of 6.5
y1 = lin_reg.predict([[6.5]])
y2 = lin_reg2.predict(poly_two.fit_transform([[6.5]]))
y3 = lin_reg3.predict(poly_three.fit_transform([[6.5]]))
y4 = lin_reg4.predict(poly_four.fit_transform([[6.5]]))
y_pred = np.append(y1, [y2, y3, y3]).reshape(-1, 1)
y_pred = np.append([[1],[2],[3],[4]], y_pred, axis=1)
