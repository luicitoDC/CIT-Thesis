#!/usr/bin/env python
# coding: utf-8

# Import the Required libraries


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Baseline Models

# The baselineModels function implements the baseline modeling of KNeighborsClassifier, DecisionTreeClassifier, GaussianNB, RandomForestClassifier, SVC and XGBClassifier. 
# This function will be used to identify the top 2 models in terms of accuracy. The top 2 models will then be subjected to hyper-parameter tuning
# and K-fold Cross Validation.

def baselineModels(train_features, train_labels, test_features, test_labels):
    
    # Evaluate KNN model
    knnTree = KNeighborsClassifier()
    knnTree = knnTree.fit(train_features, train_labels)
    
    y_pred = knnTree.predict(test_features)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_labels, predictions)
    print("KNN Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # Evaluate Decision Tree model
    decTree = DecisionTreeClassifier(random_state=0)
    decTree = decTree.fit(train_features, train_labels)
    
    y_pred = decTree.predict(test_features)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_labels, predictions)
    print("Decision Tree Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # Evaluate Naive Bayes model
    nBayes = GaussianNB()
    nBayes = nBayes.fit(train_features, train_labels)
    
    y_pred = nBayes.predict(test_features)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_labels, predictions)
    print("Naive Bayes Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # Evaluate Random Forest model
    rf = RandomForestClassifier(n_estimators=20, random_state=0)
    rf = rf.fit(train_features, train_labels)
    
    y_pred = rf.predict(test_features)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_labels, predictions)
    print("Random Forest Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # Evaluate SVM model
    svc = SVC(gamma = 'auto', random_state=0)
    svc = svc.fit(train_features, train_labels)
    
    y_pred = svc.predict(test_features)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_labels, predictions)
    print("SVC Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # Evaluate XGBoost model
    xgb = XGBClassifier()
    xgb.fit(train_features, train_labels)
    y_pred = xgb.predict(test_features)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_labels, predictions)
    print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))

# Hyper-parameter Tuning and K-Fold Cross Validation
    
# Based on my experience, Random Forest and XGBoost are the best performing 
# classification algorithms but if the results from the baseline models
# show different top 2 algorithms then these top 2 algorithms will be optimized.
    
# The optimizeRandomForest function implements the hyper-parameter tuning and
# 5-Fold Cross Validation of the Random Forest Classifier
def optimizeRandomForest(X_train, Y_train, X_test, Y_test):     
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 300]
    }
    
    # Create a based model
    rf = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 5, n_jobs = -1, verbose = 2)


    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)

    y_pred = np.round(grid_search.predict(X_test),0)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(Y_test, predictions)
    print("Random Forest Accuracy: %.2f%%" % (accuracy * 100.0))


# The optimizeXGBoosting function implements the hyper-parameter tuning and
# 5-Fold Cross Validation of the Extreme Gradient Boosting Classifier
def optimizeXGBoosting(X_train, Y_train, X_test, Y_test):
    # grid search
    model = XGBClassifier()
    n_estimators = range(50, 400, 50)
    param_grid = dict(n_estimators=n_estimators)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)

    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)
  
    y_pred = np.round(grid_search.predict(X_test),0)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(Y_test, predictions)
    print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))



