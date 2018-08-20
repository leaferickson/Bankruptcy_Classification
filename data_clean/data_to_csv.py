#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 19:08:34 2018

@author: leaferickson
"""

from scipy.io import arff
import pandas as pd
import numpy as np
import re

data = arff.loadarff('../raw_data/1year.arff')
df = pd.DataFrame(data[0])

#df["class"].str()
#df["class"] = df["class"].apply(pd.DataFrame.to_string(df["class"]))
#slize = slice(1,7027,2)
#ok = str(df["class"]).split("\'")
#ok = slize(ok)

for index, row in df.iterrows():
    new_row = re.sub('[^0-9]','', str(row["class"]))
    df.loc[index,"class"] = new_row

print(df["class"].sum())
df["class"] = df["class"].apply(pd.to_numeric)
df.dtypes

df.to_csv("../data.csv")