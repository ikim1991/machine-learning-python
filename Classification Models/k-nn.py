# K-Nearest Neighbors

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
dataset = pd.read_csv('./dataset/Social_Network_Ads.csv')
X = dataset.iloc[:,[2, 3]].values
y = dataset.iloc[:,-1].values

# Split Dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit and Predict K-NN Model from scikit learn
from sklearn.neighbors import KNeighborsClassifier as knn
classifier = knn(n_neighbors = 5)
classifier.fit(X_train, y_train)

# Predict Test Set Results
y_pred = classifier.predict(X_test)

# Generate Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true = y_test, y_pred = y_pred)

# Visualizing our Training set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
# PLotting the Decision Boundary by using a contour plot and our classifier model to separate by pixel
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.01), np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Plot our Training Dataset and add color based on label
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red','green'))(i), label = j, linewidths=0.5, s=9, edgecolors='black')
plt.title('K-NN Training Set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

# Visualizing our Test set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
# PLotting the Decision Boundary by using a contour plot and our classifier model to separate by pixel
X1, X2 = np.meshgrid(np.arange(X_set[:,0].min()-1, X_set[:,0].max()+1, 0.01), np.arange(X_set[:,1].min()-1, X_set[:,1].max()+1, 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# Plot our Training Dataset and add color based on label
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red','green'))(i), label = j, linewidths=0.5, s=9, edgecolors='black')
plt.title('K-NN Test Set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
