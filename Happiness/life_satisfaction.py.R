#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
univariate regression problem:
    predicting life satisfaction based on just one feature, the GDP per capita

Created on Fri Aug 23 18:23:12 2024

@author: king
"""

import pandas as pd
import matplotlib 
import numpy as np
import sklearn
import matplotlib.pyplot as plt

def prepare_country_stats(oecd_bli, gdp_per_capita):
    # Ensure the correct column names are used
    if 'Indicator' in oecd_bli.columns:
        oecd_bli = oecd_bli[oecd_bli["Indicator"] == "Life satisfaction"]
        oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    else:
        raise ValueError("Expected column 'Indicator' not found in oecd_bli data frame.")
    
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    
    # Merge the dataframes
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    
    return full_country_stats[["GDP per capita", "Life satisfaction"]]



# Load data
oecd_bill = "/Users/king/Documents/ML/Happiness/oecd_bli_2015.csv"
gdp = "/Users/king/Documents/ML/Happiness/gdp_per_capita.csv"

oecd_bli = pd.read_csv(oecd_bill, thousands=",")
gdp_per_capita =pd.read_csv(gdp, thousands=",", delimiter="\t", encoding="latin1", na_values="n/a")

print(oecd_bli)
print(gdp_per_capita)



# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]



# Visualize the data
country_stats.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
plt.show()

# Select a model
lin_reg_model = sklearn.linear_model.LinearRegression()

# Train the model
lin_reg_model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]] # Cyprus' GDP per capita print(lin_reg_model.predict(X_new)) # outputs [[ 5.96242338]]



"""
Replacing the Linear Regression model with k-Nearest Neighbors regression in 
the previous code is as simple as replacing this line:
    
clf = sklearn.linear_model.LinearRegression()

# with this one:
    
clf = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
"""



