#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:37:36 2018

@author: leaferickson
"""

import pandas as pd
import numpy as np
import re

col_df = pd.read_csv("column_names.txt", header = None)
for index, row in col_df.iterrows():
    new_row = str(row[0]).split(" ")[1:]
    final_row = ""
    for element in new_row:
        final_row = final_row + element + " "
#    new_row = new_row[:3].strip()
#    col_df.loc[index,0] = new_row + " NUMERIC,"

temp = pd.DataFrame(["Y NUMERIC"])
col_df = col_df.append(temp)
col_df.to_csv("sample.csv")
print(col_df)