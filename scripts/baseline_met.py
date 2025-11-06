#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 15:15:02 2025

@author: jane
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error

from preprocessing import remove_outliers

#clean data and make soil prediction------------------------------------------
path_soil = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/soil/'
features_train_soil = pd.read_pickle(path_soil + 'features_train_soil.p')
target_train_soil = pd.read_pickle(path_soil + 'target_train_soil.p')
features_test_soil = pd.read_pickle(path_soil + 'features_test_soil.p')
target_test_soil = pd.read_pickle(path_soil + 'target_test_soil.p')
target_test_soil_pred = pd.read_pickle(path_soil + 'target_test_pred.p')

path_met = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/met/'
features_train_met = pd.read_pickle(path_met + 'features_train_concat.p')
target_train_met = pd.read_pickle(path_met + 'target_train_concat.p')
features_test_met = pd.read_pickle(path_met + 'features_test_concat.p')
target_test_met = pd.read_pickle(path_met + 'target_test_concat.p')

features_soil = pd.concat([features_train_soil, features_test_soil])
model_soil = pickle.load(open('/home/jane/Documents/Weiterbildung/DPP/portfolio_project/scripts/trained_model_soil.p', 'rb'))
features_train_soil, features_train_met, target_train_soil, target_train_met = remove_outliers(features_train_soil,
                                                                                               features_train_met,
                                                                                               target_train_soil, 
                                                                                               target_train_met)


#run model on all soil data, including outliers, because we need this later
#for testing meteorological data that includes outlying fips
target_soil_pred = model_soil.predict(features_soil)
target_soil_pred = pd.DataFrame(target_soil_pred, index = features_soil.index)

#met prediction----------------------------------------------------------------

target_soil_pred_ext_for_train = []
target_soil_pred_ext_for_test = []

for fips in features_train_met.fips.unique():
    target_soil_pred_ext_for_train.extend([target_soil_pred.loc[fips, 0]]*len(features_train_met[features_train_met['fips'] == fips]))

for fips in features_test_met.fips.unique():   
    target_soil_pred_ext_for_test.extend([target_soil_pred.loc[fips, 0]]*len(features_test_met[features_test_met['fips'] == fips]))

features_train_met['pred_drought'] = target_soil_pred_ext_for_train
features_test_met['pred_drought'] = target_soil_pred_ext_for_test

features_train_met['month'] = features_train_met.date.dt.month.astype('int')
features_train_met['day'] = features_train_met.date.dt.day.astype('int')
features_train_met['year'] = features_train_met.date.dt.year.astype('int')
features_train_met = features_train_met.drop('date', axis = 1)

features_test_met['month'] = features_test_met.date.dt.month.astype('int')
features_test_met['day'] = features_test_met.date.dt.day.astype('int')
features_test_met['year'] = features_test_met.date.dt.year.astype('int')
features_test_met = features_test_met.drop('date', axis = 1)


base_lasso = Lasso()
base_ridge = Ridge()
base_EN = ElasticNet()
base_lin = LinearRegression()
base_SVR = LinearSVR()

base_dict =  {'Lasso' : base_lasso, 'Ridge' : base_ridge, 
              'Elastic Net' : base_EN, 'Linear' : base_lin,
              'SVR' : base_SVR}

for model in base_dict.keys():
    print(model)
    base_dict[model].fit(features_train_met, target_train_met)
    target_pred_met = base_dict[model].predict(features_test_met)
    mse = mean_squared_error(target_pred_met, target_test_met)
    print('Mean Squared Error for Baseline Model {} Regression is: {}'.format(model, str(mse)))
    target_pred_met = pd.DataFrame(target_pred_met, index = target_test_met.index)
    plt.plot(list(range(len(target_test_met))), target_test_met, '.')
    plt.plot(list(range(len(target_pred_met))), target_pred_met, '.')
    plt.title('Testing Data and Prediction for Model {}'.format(model))
    plt.show()