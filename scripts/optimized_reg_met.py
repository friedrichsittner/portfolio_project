#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:04:06 2025

@author: jane
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
'''
Find optimal model using Grid Search
'''

def optimize_linear_models(region, save = False):
    
    assert region in ['west', 'east', 'south', 'midwest'], 'Not a valid region. Choose from west, south, east, midwest.'

    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/met/'
    features_train = pd.read_pickle(path + 'features_train_met_prep_{}.p'.format(region))
    target_train = pd.read_pickle(path + 'target_train_met_prep_{}.p'.format(region))
    features_test = pd.read_pickle(path + 'features_test_met_prep_{}.p'.format(region))
    target_test = pd.read_pickle(path + 'target_test_met_prep_{}.p'.format(region))
    
    base_linear = LinearRegression()
    base_lasso = Lasso()
    base_ridge = Ridge()
    base_EN = ElasticNet()
    base_SVR = LinearSVR()
    
    search_space_linear = {'pca__n_components' : [0.8, 0.9, 0.95, 1]}
    search_space_ridge = search_space_linear.copy()
    search_space_ridge['Ridge__alpha'] = np.geomspace(0.01, 1000, 10)
    search_space_lasso = search_space_linear.copy()
    search_space_lasso['Lasso__alpha'] = np.geomspace(0.01, 1000, 10)
    search_space_EN = search_space_linear.copy()
    search_space_EN['ElasticNet__alpha'] = np.geomspace(0.01, 1000, 10)
    search_space_EN['ElasticNet__l1_ratio'] = np.linspace(0, 1, 10)
    search_space_SVR = search_space_linear.copy()
    search_space_SVR['SVR__C'] = np.geomspace(0.001, 1000, 15)
    
    model_dict =  {'Lasso' : [base_lasso, search_space_lasso],
                   'Linear' : [base_linear, search_space_linear],
                   'Ridge' : [base_ridge, search_space_ridge],
                  'ElasticNet' : [base_EN, search_space_EN],
                  'SVR' : [base_SVR, search_space_SVR]}
    
    for name, (model, search_space) in model_dict.items():
        pipeline = Pipeline([('std', StandardScaler()),
                             ('pca', PCA()),
                             (name, model)])
    
        grid_search = GridSearchCV(estimator = pipeline,
                               param_grid = search_space,
                               scoring = 'neg_mean_squared_error',
                               cv = 5,
                               n_jobs = -1)
    
        grid_search.fit(features_train, np.ravel(target_train))
        print('Results for Model {}'.format(name))
        print(grid_search.best_params_)
        print(grid_search.best_score_)
    
        target_pred = grid_search.predict(features_test)
        mse = mean_squared_error(target_pred, target_test)
        print('Mean Squared Error for Model optimized {} Regression is: {}'.format(name, str(mse)))
        target_pred = pd.DataFrame(target_pred, index = target_test.index)
        fig = plt.figure()
        fig.plot(list(range(len(target_test))), target_test, '.', label = 'Test Data')
        fig.plot(list(range(len(target_pred))), target_pred, '.', label = 'Prediction')
        fig.title('Testing Data and Prediction for Optimized Ridge Regression')
        fig.legend()
        if save == True:
            plt.savefig('optimized_{}_{}.png'.format(name, region))
            
        return
    
if __name__ == '__main__':
    region = input('Region: ')
    save = input('Do you want graphs: ')
    optimize_linear_models(region, save)
    