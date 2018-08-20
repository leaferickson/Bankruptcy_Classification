# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
data = pd.read_csv("../data.csv")
cols = pd.read_csv("sample.csv")

cols.columns
data = data.loc[:,"Attr1":"Attr64"]
data.columns = cols.columns
data.dropna()

model = pickle.load(open("model.pkl", "rb"))
prediction = model.predict(data.dropna())




means = pd.DataFrame(data.mean()).transpose()
stdevs = pd.DataFrame(data.std()).transpose()
pickle.dump(means, open("means.pkl", "wb"))
pickle.dump(stdevs, open("stdevs.pkl", "wb"))





means = np.array(pickle.load(open("means.pkl", "rb")))
stdevs = np.array(pickle.load(open("stdevs.pkl", "rb")))
model = pickle.load(open("model.pkl", "rb"))
data = pd.read_csv("test.csv")
data = np.array(data)
scaled_data = (data - means) / stdevs
predictions = model.predict(scaled_data)
dat2 = pd.DataFrame(data)
dat2["prediction"] = predictions

dat2.to_csv("results.csv")