#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:04:06 2025

@author: jane
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

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

def optimize_linear_models(region, model_name, save = False):
    
    assert region in ['west', 'east', 'south', 'midwest'], 'Not a valid region. Choose from west, south, east, midwest.'
    assert model_name in ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'LinearSVR']
    if save in ['yes', 'Yes', 'y', 'Y']:
        save = True
        
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/met/'
    features_train = pd.read_pickle(path + 'features_train_met_prep_{}.p'.format(region))
    target_train = pd.read_pickle(path + 'target_train_met_prep_{}.p'.format(region))
    features_test = pd.read_pickle(path + 'features_test_met_prep_{}.p'.format(region))
    target_test = pd.read_pickle(path + 'target_test_met_prep_{}.p'.format(region))
    
    if model_name == 'LinearRegression':
        model = LinearRegression()
        search_space = {'pca__n_components' : [0.8, 0.9, 0.95, 1]}
    elif model_name == 'Lasso':
        model = Lasso()
        search_space = {'pca__n_components' : [0.8, 0.9, 0.95, 1],
                        'Lasso__alpha' : np.geomspace(0.01, 1000, 10)}
    elif model_name == 'Ridge':
        model = Ridge()
        search_space = {'pca__n_components' : [0.8, 0.9, 0.95, 1],
                        'Ridge__alpha' : np.geomspace(0.01, 1000, 10)}
    elif model_name == 'ElasticNet':
        model = ElasticNet()
        search_space = {'pca__n_components' : [0.8, 0.9, 0.95, 1],
                        'ElasticNet__alpha' : np.geomspace(0.01, 1000, 10),
                        'ElasticNet__l1_ratio' : np.linspace(0, 1, 10)}
    else:
        model = LinearSVR()
        search_space = {'pca__n_components' : [0.8, 0.9, 0.95, 1],
                        'LinearSVR__C' : np.geomspace(0.001, 1000, 15)}
        
    pipeline = Pipeline([('std', StandardScaler()),
                         ('pca', PCA()),
                         (model_name, model)])
    
    grid_search = GridSearchCV(estimator = pipeline,
                           param_grid = search_space,
                           scoring = 'neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
    
    grid_search.fit(features_train, np.ravel(target_train))
    print('Results for Model {}'.format(model_name))
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    
    target_pred = grid_search.predict(features_test)
    mse = mean_squared_error(target_pred, target_test)
    print('Mean Squared Error for Model optimized {} Regression is: {}'.format(model_name, str(mse)))
    target_pred = pd.DataFrame(target_pred, index = target_test.index)
    fig, ax = plt.subplots()
    ax.plot(list(range(len(target_test))), target_test, '.', label = 'Test Data')
    ax.plot(list(range(len(target_pred))), target_pred, '.', label = 'Prediction')
    ax.set(title = 'Testing Data and Prediction for Optimized Ridge Regression')
    ax.legend()
    if save == True:
        plt.savefig('optimized_{}_{}.png'.format(model_name, region))
    with open('optimized_reg_results.txt', 'a') as f:
        f.write('Run at {}\n'.format(str(datetime.datetime.now())))
        f.write('Region: ' + str(region) + '\n')
        f.write('Results for Model {}\n'.format(model_name))
        f.write(str(grid_search.best_params_) + '\n')
        f.write(str(grid_search.best_score_) + '\n')
        f.write('Model {} for Region {} yields a Mean Squared Error of {}\n\n'.format(model_name, region, str(mse)))
    return
    

def optimize_linear_models_wo_pca(region, model_name, save = False):
    
    assert region in ['west', 'east', 'south', 'midwest'], 'Not a valid region. Choose from west, south, east, midwest.'
    assert model_name in ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'LinearSVR']
    if save in ['yes', 'Yes', 'y', 'Y']:
        save = True
        
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/met/'
    features_train = pd.read_pickle(path + 'features_train_met_prep_{}.p'.format(region))
    target_train = pd.read_pickle(path + 'target_train_met_prep_{}.p'.format(region))
    features_test = pd.read_pickle(path + 'features_test_met_prep_{}.p'.format(region))
    target_test = pd.read_pickle(path + 'target_test_met_prep_{}.p'.format(region))
    
    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'Lasso':
        model = Lasso()
        search_space = {'Lasso__alpha' : np.geomspace(0.01, 1000, 10)}
    elif model_name == 'Ridge':
        model = Ridge()
        search_space = {'Ridge__alpha' : np.geomspace(0.01, 1000, 10)}
    elif model_name == 'ElasticNet':
        model = ElasticNet()
        search_space = {'ElasticNet__alpha' : np.geomspace(0.01, 1000, 10),
                        'ElasticNet__l1_ratio' : np.linspace(0, 1, 10)}
    else:
        model = LinearSVR()
        search_space = {'LinearSVR__C' : np.geomspace(0.001, 1000, 15)}
        
    #features_test = wind_temp_pca(features_test, target_test, n_comp = 2)
    wind_temp_pca = pickle.load('wind_temp_pca.p')
    n_components = (len(wind_temp_pca.named_transformers_['pca_temp'].explained_variance_ratio_) +
                    len(wind_temp_pca.named_transformers_['pca_wind'].explained_variance_ratio_))
    pca_names = ['PCA_comp_' + str(i) for i in range(n_components)]
    remaining_names = features_test.columns[wind_temp_pca._remainder[2]]
    features_test = pd.DataFrame(wind_temp_pca.transform(features_test),
                                  columns = list(pca_names)+ list(remaining_names),
                                  index = target_test.index)
    
    
    pipeline = Pipeline([('std', StandardScaler()),
                         (model_name, model)])
    
    grid_search = GridSearchCV(estimator = pipeline,
                           param_grid = search_space,
                           scoring = 'neg_mean_squared_error',
                           cv = 5,
                           n_jobs = -1)
    
    grid_search.fit(features_train, np.ravel(target_train))
    print('Results for Model {}'.format(model_name))
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(features_train.columns)
    target_pred = grid_search.predict(features_test)
    mse = mean_squared_error(target_pred, target_test)
    print('Mean Squared Error for Model optimized {} Regression is: {}'.format(model_name, str(mse)))
    target_pred = pd.DataFrame(target_pred, index = target_test.index)
    fig, ax = plt.subplots()
    ax.plot(list(range(len(target_test))), target_test, '.', label = 'Test Data')
    ax.plot(list(range(len(target_pred))), target_pred, '.', label = 'Prediction')
    ax.set(title = 'Testing Data and Prediction for Optimized Ridge Regression')
    ax.legend()
    if save == True:
        plt.savefig('optimized_{}_{}.png'.format(model_name, region))
    with open('optimized_reg_results.txt', 'a') as f:
        f.write('Run at {}\n'.format(str(datetime.datetime.now())))
        f.write('Region: ' + str(region) + '\n')
        f.write('Results for Model {}\n'.format(model_name))
        f.write(str(grid_search.best_params_) + '\n')
        f.write(str(grid_search.best_score_) + '\n')
        f.write('Model {} for Region {} yields a Mean Squared Error of {}\n\n'.format(model_name, region, str(mse)))
    return

if __name__ == '__main__':
    region = input('Region: ')
    model_name = input('Model: ')
    save = input('Do you want graphs: ')
    optimize_linear_models_wo_pca(region, model_name, save)
    