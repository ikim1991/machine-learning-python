# XGBoost

## Summary

XGBoost is a gradient boosting algorithm which utilizes an ensemble of decision trees. It is a powerful model that is fast and high in performance. In this example we will predict the churn rate of customers of a bank found in the [Churn Modelling dataset](./dataset/Churn_Modelling.csv) using the XGBoost algorithm.

## XGBoost

To use the XGBoost classification algorithm, make sure to install the library onto your machine. After installing the library we can utilize the XGBClassifier class from the xgboost library. Feature scaling is not required for the XGBoost classifier as it uses decision trees in gradient boosting. After preprocessing the data XGBoost can be called as shown below.

```Python
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
```

Using the k-fold cross validation method, we can see that we get a relatively good model with an average accuracy of 86.3% and a standard deviation of 1.1%.
