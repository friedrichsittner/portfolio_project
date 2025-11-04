#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:04:06 2025

@author: jane
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from data_processing import remove_outliers

def ridge_model(features_train, features_test, target_train, target_test):
    
    '''
    Fits a Ridge Regression Model and predicts target data
    Params:
        - features_train (pd.DataFrame): preprocessed training set
        - features_test (pd.DataFrame): testing set
        - target_train (pd.DataFrame): preprocessed training target
        - target_test (pd.DataFrame): testint target
        
    Returns:
        - pipeline: pipeline of Scaler, PCA, and Ridge Regressor fit to training data
        - target_pred: predicted testing target
    '''
    
    pipeline = Pipeline([('std', StandardScaler()),
                         ('pca', PCA(n_components = 0.95)),
                         ('lin_model', Ridge(alpha = 77))])
    
    
    
    target_pred = pipeline.predict(features_test)
    mse = mean_squared_error(target_pred, target_test)
    print('Mean Squared Error for Baseline Model optimized Ridge Regression is: {}'.format(str(mse)))
    target_pred = pd.DataFrame(target_pred, index = target_test.index)
    return pipeline, target_pred

if __name__ == '__main__':
    
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/soil/'
    features_train = pd.read_pickle(path + 'features_train_soil.p')
    target_train = pd.read_pickle(path + 'target_train_soil.p')
    features_test = pd.read_pickle(path + 'features_test_soil.p')
    target_test = pd.read_pickle(path + 'target_test_soil.p')

    features_train, target_train = remove_outliers(features_train, target_train)
    
    ridge_model(features_train, features_test, target_train, target_test)