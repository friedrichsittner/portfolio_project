#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 11:41:06 2025

@author: jane
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error

from preprocessing_soil import remove_outliers

path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/soil/'
features_train = pd.read_pickle(path + 'features_train_soil.p')
target_train = pd.read_pickle(path + 'target_train_soil.p')
features_test = pd.read_pickle(path + 'features_test_soil.p')
target_test = pd.read_pickle(path + 'target_test_soil.p')

features_train, target_train = remove_outliers(features_train, target_train)

base_lasso = Lasso()
base_ridge = Ridge()
base_EN = ElasticNet()
base_lin = LinearRegression()

base_dict =  {'Lasso' : base_lasso, 'Ridge' : base_ridge, 
              'Elastic Net' : base_EN, 'Linear' : base_lin}

for model in base_dict.keys():
    print(model)
    base_dict[model].fit(features_train, target_train)
    target_pred = base_dict[model].predict(features_test)
    mse = mean_squared_error(target_pred, target_test)
    print('Mean Squared Error for Baseline Model {} Regression is: {}'.format(model, str(mse)))
    target_pred = pd.DataFrame(target_pred, index = target_test.index)
    plt.plot(list(range(len(target_test))), target_test, '.')
    plt.plot(list(range(len(target_pred))), target_pred, '.')
    plt.title('Testing Data and Prediction for Model {}'.format(model))
    plt.show()
