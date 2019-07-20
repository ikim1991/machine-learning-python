# Machine Learning: Classification Models on Python

## Summary

In previous examples, we used regression models to predict real values. In a classification model we predict a categorical output. Much like regression models, classification models have linear and non-linear models based on its decision boundary.

In a classification problem, we would categorize the classes by mapping numerical values to the labels. In a binary classification we would denote 0 as the negative class and 1 as the positive class. In other words, our data points will either be labeled as Y = 0 or Y = 1.

We will look at a customer segmentation example to find out the targeted customers for SUVs. The [Social Network Ads dataset](./dataset/Social_Network_Ads.csv) contains a customer database by age and estimated salary, and whether they purchased the SUV or not. The positive class (Y=1) for this example is classified as the customers who bought the SUV.

The following classification models will be examined:
  - Logistic Regression
  - K-Nearest-Neighbor (K-NN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest

All models follows the same preprocessing and visualization steps listed below.

Using Pandas we load the [Social Network Ads dataset](./dataset/Social_Network_Ads.csv).

```Python
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
dataset = pd.read_csv('./dataset/Social_Network_Ads.csv')
X = dataset.iloc[:,[2, 3]].values
y = dataset.iloc[:,-1].values
```

Then we preprocess the data by splitting them into train/test sets and performing feature scaling.

```Python
# Split Dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```

To evaluate the performance of each model, we will use the confusion matrix from the scikit-learn metrics library. The confusion matrix takes in 2 arguments, the predicted values, y_pred and compares it to the actual values, y_true. This allows us to measure such metrics as accuracy, precision, recall, and the F-Score. For this example we only really need to focus on the accuracy metrics. We would focus more on precision, recall, and F-Score if we were dealing with an uneven class distribution.

```Python
# Generate Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true = y_test, y_pred = y_pred)
```

The confusion matrix outputs a matrix in the following format:

[ TP FP ]
[ FN TN ]

- A True Positive (TP) occurs when the model correctly predicts a positive class
- A True Negative (TN) occurs when the model correctly predicts a negative class
- A False Positive (FP) occurs when the model incorrectly predicts a positive class when in fact it is actually negative
- A False Negative (TN) occurs when the model incorrectly predicts a negative class when in fact it is actually positive

We visualize our results and the decision boundaries of our models using matplotlib and its contour plots.

```Python
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
```

## Logistic Regression Classification Model

A logistic regression transforms its outputs using the sigmoid function to return a probability value between 0 and 1, which can then be mapped to different classes. Our model will predict the probability of whether the classification falls under Y = 0 or Y = 1. We would need to determined the threshold value of our model, by default we set this to 0.5, meaning if there is over a 50% probability of the prediction being true, we assume it to be true. If our prediction is greater than 0.5 we assume that Y = 1 and if our prediction is less than 0.5 we assume that Y = 0.

We fit and train the logistic regression model using the LogisticRegression class from the scikit-learn linear_model library.

```Python
# Fit and Predict Logistic Regression Model from scikit learn
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predict Test Set Results
y_pred = classifier.predict(X_test)
```

Next we predict the results using the test set and evaluate the performance of our model using the confusion matrix to get the following output:

[65	 3]
[8	24]

We can calculate the accuracy of our model to be 89%.

Using matplotlib we can visualize the results below. We can see that the logistic regression is a linear model from its decision boundary.

![Training Set Results](./lr-training)
![Test Set Results](./lr-test)

## K-Nearest-Neighbor Classification Model

We will now explore the K-Nearest-Neighbor (K-NN) model. The first step in using the K-NN is to determine the number of K neighbors, or the number of the closest data points the model looks at. By default the number of K neighbors is 5. This means that the model will predict based on the 5 closest data points by Euclidean distances and classify based on the majority. The K-NN model is a non-linear model and the classes are separated into different regions. By default, we use Euclidean distances but we could also use other types of distances if we wanted to.

We fit and train the K-NN model using the KNeighborsClassifier class from the scikit-learn neighbors library. We need to define the n_neighbors as an argument which determines the number of K neighbors the model looks at. By default we chose this value to be 5.

```Python
# Fit and Predict K-NN Model from scikit learn
from sklearn.neighbors import KNeighborsClassifier as knn
classifier = knn(n_neighbors = 5)
classifier.fit(X_train, y_train)

# Predict Test Set Results
y_pred = classifier.predict(X_test)
```

Next we predict the results using the test set and evaluate the performance of our model using the confusion matrix to get the following output:

[64	 4]
[3	29]

We can calculate the accuracy of our model to be 93%.

Using matplotlib we can visualize the results below. We can see that the k-nn is a non-linear model from its decision boundary.

![Training Set Results](./knn-training)
![Test Set Results](./knn-test)

## Support Vector Machine (SVM) Classification Model

The SVM can be either linear or non-linear depending on the type of kernel we use. We can achieve a non-linear SVM model by mapping our vector of features to a higher dimensionality. The SVM algorithm finds the optimal decision boundary of our data using specific supporting vectors and their Maximum Margin Hyperplane. In other words, the SVM algorithm looks at data points closest to the decision boundary and finds the largest margin within these data points.

There are several types of kernels we can consider when using the SVM algorithm. The type of kernel to use will depend on the features, distribution of data and the decision boundary. Some common types of kernels are,
  - Linear
  - Gaussian RBF Kernel
  - Sigmoid Kernel
  - Polynomial Kernel

We fit and train the SVM model using the SVC class from the scikit-learn svm library. We need to define the kernel type as an argument for the model to use. By default this is 'rbf' or the Gaussian RBF Kernel.

```Python
# Fit and Predict SVM Model from scikit learn
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predict Test Set Results
y_pred = classifier.predict(X_test)
```

Next we predict the results using the test set and evaluate the performance of our model using the confusion matrix to get the following output:

[64	 4]
[3	29]

We can calculate the accuracy of our model to be 93%.

Using matplotlib we can visualize the results below. As we can see, the SMV is a non-linear model from its decision boundary. We can achieve a linear model if we were to use the linear kernel type for the SVM model.

![Training Set Results](./svm-training)
![Test Set Results](./svm-test)

## Decision Tree Classification Model

We previously explored a Decision Tree regression model to predict for real values. For this example we will now use the Decision Tree to predict a classification problem. In a supervised learning problem, we have a labeled set of data. The algorithm splits the data into multiple segments based on its labels.

We fit and train the Decision Tree model using the DecisionTreeClassifier class from the scikit-learn tree library. We need to define the criterion as an argument for the model to use. The criterion function measures the quality of a split in the decision tree. We use the 'entropy' criterion as our optimizing function for information gain.

```Python
# Fit and Predict Decision Tree Model from scikit learn
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predict Test Set Results
y_pred = classifier.predict(X_test)
```

Next we predict the results using the test set and evaluate the performance of our model using the confusion matrix to get the following output:

[62	 6]
[3	29]

We can calculate the accuracy of our model to be 91%.

Using matplotlib we can visualize the results below. We can see that the decision tree is a non-linear model from its decision boundary.

![Training Set Results](./decision-tree-training)
![Test Set Results](./decision-tree-test)

## Random Forest Classification Model

As we discussed before, Random Forest is a type of ensemble learning that utilizes a network of decision trees. To summarize, in a random forest we select at random a subset of K data points from the training set. Then using the subset of K data points we would build a decision tree model. We would determine the N number of trees to utilize and repeat this process N number of times. A new data point can be predicted and classified based on the predictions of the majority.

We fit and train the Random Forest model using the RandomForestClassifier class from the scikit-learn ensemble library. We need to define the n_estimators and criterion as arguments for the model to use. The n_estimators is the N number of decision trees the random forest uses and the criterion function measures the quality of a split in the decision tree. We will use 100 n_estimators, and the 'entropy' criterion as our optimizing function for information gain.

```Python
# Fit and Predict Random Forest Model from scikit learn
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predict Test Set Results
y_pred = classifier.predict(X_test)
```

Next we predict the results using the test set and evaluate the performance of our model using the confusion matrix to get the following output:

[63	 5]
[4	28]

We can calculate the accuracy of our model to be 91%.

Using matplotlib we can visualize the results below. We can see that the random forest model is a non-linear model from its decision boundary.

![Training Set Results](./random-forest-training)
![Test Set Results](./random-forest-test)

## Closing Thoughts

For this particular example of customer segmentation, we can see that a non-linear model fits our dataset the best. Specifically the Gaussian Kernel SVM model had the highest predictive accuracy while the linear model of logistic regression had the lowest. Using model selection we would determine the best model to use in a given situation.
