#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:04:06 2025

@author: jane
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from preprocessing_soil import remove_outliers

'''
Find optimal model using Grid Search
'''

path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/soil/'
features_train = pd.read_pickle(path + 'features_train_soil.p')
target_train = pd.read_pickle(path + 'target_train_soil.p')
features_test = pd.read_pickle(path + 'features_test_soil.p')
target_test = pd.read_pickle(path + 'target_test_soil.p')

features_train, target_train = remove_outliers(features_train, target_train)

pipeline = Pipeline([('std', StandardScaler()),
                     ('pca', PCA()),
                     ('lin_model', Ridge())])

search_space = {'pca__n_components' : [0.8, 0.9, 0.95, 1],
                'lin_model__alpha' : np.geomspace(0.01, 1000, 10)}
#                'lin_model__l1_ratio' : np.linspace(0, 1, 20)}

grid_search = GridSearchCV(estimator = pipeline,
                       param_grid = search_space,
                       scoring = 'neg_mean_squared_error',
                       cv = 5,
                       n_jobs = -1)

grid_search.fit(features_train, target_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

target_pred = grid_search.predict(features_test)
mse = mean_squared_error(target_pred, target_test)
print('Mean Squared Error for Baseline Model optimized Ridge Regression is: {}'.format(str(mse)))
target_pred = pd.DataFrame(target_pred, index = target_test.index)
plt.plot(list(range(len(target_test))), target_test, '.')
plt.plot(list(range(len(target_pred))), target_pred, '.')
plt.title('Testing Data and Prediction for Optimized Ridge Regression')
plt.show()