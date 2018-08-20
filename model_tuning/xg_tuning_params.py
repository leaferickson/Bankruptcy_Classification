# -*- coding: utf-8 -*-

#np.sum(data.values >= np.finfo(np.float64).max)
#np.sum(data.values >= np.finfo(np.float32).max)#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:20:02 2018

@author: leaferickson
"""
#np.isnan(data.values.any())
#data.dropna(how = "any")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
import imblearn.over_sampling
import matplotlib.pyplot as plt
data = pd.read_csv("../data.csv")

#data["value"] = data.groupby("name").transform(lambda x: x.fillna(x.mean()))
#X = data.loc[:,"Attr1":"Attr64"]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45, stratify = y)
data_train, data_test = train_test_split(data, test_size=0.3, random_state=45, stratify = data["class"])

data_test.groupby("class").mean()
X_train = data_train.transform(lambda x: x.fillna(x.mean())).loc[:,"Attr1":"Attr64"]
y_train = data_train["class"]
X_test = data_test.transform(lambda x: x.fillna(x.mean())).loc[:,"Attr1":"Attr64"]
y_test = data_test["class"]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


ROS = imblearn.over_sampling.RandomOverSampler(\
                                               ratio={0:4728,1:190*24}, \
                                               random_state=42) 
    
X_train, y_train = ROS.fit_sample(X_train, y_train)


# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
learning_rate = [0.1, 0.05, 0.01]
# Ratio of columns used at each split
colsample_bylevel = [1, 0.75, 0.5, 0.25]
# Impurity threshold change under which to no longer split
gamma = [0, 0.02, 0.05, 0.1]
# Create the random grid
random_grid = {'max_depth': max_depth,
               'learning_rate': learning_rate,
               'colsample_bylevel': colsample_bylevel,
               'gamma': gamma}

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = XGBClassifier(), scoring = 'roc_auc', param_distributions = random_grid, n_iter = 72, cv = 3, verbose=2, random_state=31, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_score_
rf_random.best_params_
rf_random.best_estimator_


rf = XGBClassifier(n_estimators = 1800, min_samples_split = 10, min_samples_leaf = 1, min_impurity_decrease = 0, max_depth = 110, random_state = 31)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
rf.score(X_test, y_test)
roc_auc_score(y_test, rf.predict(X_test))



from sklearn.metrics import confusion_matrix
import itertools

plt.figure(dpi=150)
cm = confusion_matrix(y_test, rf.predict(X_test))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xticks([0,1],["Operating", "Bankrupt"])
plt.yticks([0,1],["Operating", "Bankrupt"])
plt.title("5 Year Bankruptcy Prediction", y = 1.05)
plt.ylabel("Reality", rotation = 0)
plt.xlabel("Prediction")
fmt = '.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
#    plt.savefig("RF_Naive")# -*- coding: utf-8 -*-

