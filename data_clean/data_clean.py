

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
from sklearn.model_selection import train_test_split
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



rf = GradientBoostingClassifier(n_estimators = 400, random_state = 31, n_jobs = -1)
rf.fit(X_train, y_train)
rf.score(X_train, y_train)
rf.score(X_test, y_test)
roc_auc_score(y_test, rf.predict(X_test))



from sklearn.metrics import confusion_matrix
import itertools

plt.figure(dpi=150)
cm = confusion_matrix(y_test, rf.predict(X_test))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks([0,1])
plt.yticks([0,1])
plt.title("Predicting Polish Bankruptcy within 5 Years")
plt.ylabel("True")
plt.xlabel("Predicted")
fmt = '.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    plt.savefig("RF_Naive")
