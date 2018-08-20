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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
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


# Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 3)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 40, num = 8)]
# Minimum number of samples required to split a node
min_samples_split = [8, 12, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Impurity threshold change under which to no longer split
min_impurity_decrease = [0, 0.02, 0.05, 0.1]
# Create the random grid
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'min_impurity_decrease': min_impurity_decrease}

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), scoring = 'roc_auc', param_distributions = random_grid, n_iter = 125, cv = 3, random_state=31, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_score_
rf_random.best_params_
rf_random.best_estimator_



#Number of Trees
n_estimators = [500, 700, 900]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(90, 110, num = 2)]
# Minimum number of samples required to split a node
min_samples_split = [25]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Impurity threshold change under which to no longer split
min_impurity_decrease = [0]
# Create the random grid
random_grid = {'max_features': max_features,
               'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'min_impurity_decrease': min_impurity_decrease}

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = GridSearchCV(estimator = RandomForestClassifier(), scoring = 'roc_auc', verbose = 2, param_grid = random_grid, cv = 3, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_score_
rf_random.best_params_
rf_random.best_estimator_



rf = RandomForestClassifier(n_estimators = 1800, min_samples_split = 25, min_samples_leaf = 1, min_impurity_decrease = 0, max_depth = 110, random_state = 31)
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
    plt.savefig("Predictions.jpg")



#Save Output
preds = rf.predict(X_train)
temp = pd.DataFrame(X_train)
temp["class_prediction"] = preds
temp["class_real"] = pd.DataFrame(y_train).reset_index(drop = True)
preds2 = rf.predict(X_test)
temp2 = pd.DataFrame(X_test)
temp2["class_prediction"] = preds2
temp2["class_real"] = pd.DataFrame(y_test).reset_index(drop = True)
frames = [temp, temp2]
data_out = pd.concat(frames)

data_out.to_csv("preds_data.csv")