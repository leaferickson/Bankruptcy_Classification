# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
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


pred_weights = {0:1, 1:1}
models = list()

knnmodel = KNeighborsClassifier()
knnmodel.fit(X_train, y_train)
models.append(knnmodel)
roc_auc_score(y_test, knnmodel.predict(X_test))

#svmodel = SVC(kernel = 'rbf', C = 1, random_state = 31)
#svmodel.fit(X_train, y_train)
#models.append(svmodel)
#roc_auc_score(y_test, svmodel.predict(X_test))
svmodel2 = SVC(kernel = 'rbf', C = 1000, random_state = 31)
svmodel2.fit(X_train, y_train)
models.append(svmodel2)
roc_auc_score(y_test, svmodel2.predict(X_test))

#logit = LogisticRegression(C = 1, class_weight = pred_weights, random_state = 31, n_jobs = -1)
#logit.fit(X_train, y_train)
#models.append(logit)
#roc_auc_score(y_test, logit.predict(X_test))
logit2 = LogisticRegression( C = 1000, class_weight = pred_weights, random_state = 31, n_jobs = -1)
logit2.fit(X_train, y_train)
models.append(logit2)
roc_auc_score(y_test, logit2.predict(X_test))

rf = RandomForestClassifier(n_estimators = 400, random_state = 31)
rf.fit(X_train, y_train)
models.append(rf)
roc_auc_score(y_test, rf.predict(X_test))

xg_boost = XGBClassifier(n_estimators = 400, random_state = 31)
xg_boost.fit(X_train, y_train)
models.append(xg_boost)
roc_auc_score(y_test, xg_boost.predict(X_test))

gradient_boost = GradientBoostingClassifier(n_estimators = 400, random_state = 31)
gradient_boost.fit(X_train, y_train)
models.append(gradient_boost)
roc_auc_score(y_test, gradient_boost.predict(X_test))


model_names = ["KNN", "SVC", "Logistic Regression", "Random Forest", "XG Boost"]
count = 0
plt.figure(figsize=(25,15))
for model in models:
    print(model)
    fpr, tpr, threshold_array = roc_curve(y_test, model.predict(X_test))
    lw = 2
    plt.plot(fpr, tpr,
             lw=lw, label= model_names[count])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tick_params(axis='both', which='major', labelsize=26)
    plt.xlabel('False Positive Rate', fontsize = 26)
    plt.ylabel('True Positive Rate', fontsize = 26, rotation = 0)
    plt.title('ROC Curves for Model Types', fontsize = 32)
    plt.legend(loc="lower right", prop={'size': 32})
    count += 1
plt.savefig('fig1.png')
plt.show()


# randomly oversample by telling it the number of samples to have in each class
ROS = imblearn.over_sampling.RandomOverSampler(\
                                               ratio={0:4728,1:190*24}, \
                                               random_state=42) 
X_train_oversample, y_train_oversample = ROS.fit_sample(X_train, y_train)



rf = RandomForestClassifier(n_estimators = 200, random_state = 31)
rf.fit(X_train_oversample, y_train_oversample)
models.append(rf)
roc_auc_score(y_test, rf.predict(X_test))

xg_boost = XGBClassifier(n_estimators = 200, random_state = 31)
xg_boost.fit(X_train_oversample, y_train_oversample)
models.append(xg_boost)
roc_auc_score(y_test, xg_boost.predict(X_test))


gradient_boost = GradientBoostingClassifier(n_estimators = 400, random_state = 31)
gradient_boost.fit(X_train_oversample, y_train_oversample)
models.append(gradient_boost)
roc_auc_score(y_test, gradient_boost.predict(X_test))
