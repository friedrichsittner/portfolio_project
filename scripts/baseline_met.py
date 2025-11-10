#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 15:15:02 2025

@author: jane
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error

def baseline_met(region, save = False):
    
    assert region in ['west', 'east', 'south', 'midwest'], 'Not a valid region. Choose from west, south, east, midwest.'
    
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/met/'
    features_train_met = pd.read_pickle(path + 'features_train_met_prep_{}.p'.format(region))
    features_test_met = pd.read_pickle(path + 'features_test_met_prep_{}.p'.format(region))
    target_train_met = pd.read_pickle(path + 'target_train_met_prep_{}.p'.format(region))
    target_test_met = pd.read_pickle(path + 'target_test_met_prep_{}.p'.format(region))

    base_lasso = Lasso()
    base_ridge = Ridge()
    base_EN = ElasticNet()
    base_lin = LinearRegression()
    base_SVR = LinearSVR()
    
    base_dict =  {'Lasso' : base_lasso, 'Ridge' : base_ridge, 
                  'Elastic Net' : base_EN, 'Linear' : base_lin,
                  'SVR' : base_SVR}
    
    with open('baseline_results.txt', 'a') as f:
        f.write('\nRun at {}\n'.format(str(datetime.datetime.now())))
    for model in base_dict.keys():
        print(model)
        base_dict[model].fit(features_train_met, target_train_met)
        target_pred_met = base_dict[model].predict(features_test_met)
        mse = mean_squared_error(target_pred_met, target_test_met)
        print('Mean Squared Error for Baseline Model {} Regression is: {}'.format(model, str(mse)))
        target_pred_met = pd.DataFrame(target_pred_met, index = target_test_met.index)
        fig = plt.figure()
        fig.plot(list(range(len(target_test_met))), target_test_met, '.', label ='Testing Data')
        fig.plot(list(range(len(target_pred_met))), target_pred_met, '.', label = 'Target Data')
        fig.title('Testing Data and Prediction for Model {}'.format(model))
        fig.legend()
        if save == True:
            plt.savefig('baseline_{}_{}.png'.format(model, region))
        plt.close(fig)
        with open('baseline_results.txt', 'a') as f:
            f.write('Model {} for Region {} yields a Mean Squared Error of {}\n'.format(model, region, str(mse)))
        
if __name__ == '__main__':
    for region in ['east', 'west', 'south', 'midwest']:
        baseline_met(region, save = False)