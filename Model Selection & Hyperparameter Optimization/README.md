# Model Selection & Hyperparameter Optimization

## Summary

In this example we will explore the process of model selection and optimizing the hyperparameters of our machine learning algorithms. To accomplish this we will utilize two well known techniques:

  - k-Fold Cross Validation
  - Grid Search

## k-Fold Cross Validation

The k-fold cross validation technique is an effective means of evaluating the bias-variance tradeoff of a machine learning model. The key goal of a machine learning model is to be able to generalize well over new datasets. A low bias and low variance means we have a good model. A high variance means that the model is too specific and will overfit the data while a high bias means that the model is too general and will underfit the data. In other words, high variance models takes too much information (noise) from the data while high bias models takes too little information from the data.

By using the k-fold cross validation technique, we can deduce if our model is suffering from a high variance or high bias problem. From the SVM example of the SUV customer segmentation problem, by doing a 80/20 split on the dataset, we concluded that the models' accuracy was 91%. By using the k-fold cross validation technique we can see how well our model generalizes. The Python code is shown below.

```Python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X =  X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
```

 We need to use the cross_val_score class from the sklearn.model_selection library. The cross_val_score class takes in 4 main arguments. The classification model (estimator), the X and Y training sets, and the number of cross validations (cv). We use a cross validation of 10, meaning the training data set was split into 10 different cross validation sets to train our model. Running the k-fold cross validation, we get an average accuracy of 90.1% and a standard deviation of 6.4%. We can conclude that our model is performing well and that it does not suffer a high variance or high bias problem. If our model were subject to high variance or high bias, we would see the average accuracy at a much lower rate, as well as large discrepancies in the standard deviation.

## Grid Search

First we used the k-fold cross validation technique to valid our model. Now we can use the grid search technique to optimize and improve our model by tuning the hyperparameters. Grid search tries every possible combination of hyperparameters and finds the best set of hyperparameters based on a performance metric. Building on the SVM model of the SUV customer segmentation example we get the following:

```Python
from sklearn.model_selection import GridSearchCV
parameters = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        ]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring='accuracy', cv = 10)
grid_search = grid_search.fit(X = X_train, y = y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
```

Using the GridSearchCV class from the sklearn.model_selection library, we first need to define a set of hyperparameters. We need to define it in a list as a set of dictionaries. As shown in the code above we look at 2 different cases. The first using a linear kernel and the second using rbf Gaussian kernel. Focusing the loss penalty parameter, C and the standard deviation parameter of the hyperplane, gamma. The GridSearchCV class takes in 4 main arguments. The classification model (estimator), the defined set of hyperparameters (param_grid), the scoring metric (scoring), and the number of cross validations to test on (cv). We then fit it on the training set and use the best_score_ and best_params_ methods to find the highest accuracy and the set of parameters that yielded the highest accuracy. In this example the highest accuracy was found at 90.3%, which was built using the hyperparameters: C = 1, gamma = 0.7, and kernel = 'rbf'.
