#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 15:55:04 2025

@author: jane
"""

import pandas as pd

def concatenate(features_train, features_test, target_train, target_test):
    '''
    Drought levels only measured every 7th day, average all data for the preceeding week
    Params:
        - features_train (pd.DataFrame): preprocessed training set
        - features_test (pd.DataFrame): testing set
        - target_train (pd.DataFrame): preprocessed training target
        - target_test (pd.DataFrame): testing target
        
    Returns:
        - features_train (pd.DataFrame): preprocessed training set, averaged weekly
        - features_test (pd.DataFrame): testing set, averaged weekly
        - target_train (pd.DataFrame): preprocessed training target, averaged weekly
        - target_test (pd.DataFrame): testing target, averaged weekly
    '''
    
    data_train_test_concat =[]
    data = {'train' : (features_train, target_train),
                    'test' : (features_test, target_test)}
    for dataset in data:
        features, target = data[dataset][0], data[dataset][1]
        target_concat = target[~target.isna()]
        features_concat = {}
        for feat in features.columns:
            feature_list = [features.loc[0, feat]]
            old_week = 0
            print('Concatenating feature {} of {}ing data.'.format(feat, dataset))
            for week in target_concat.index[1:]:
                feature_list.append(features.loc[old_week + 1:week + 1, feat].mean())
                old_week = week
            features_concat[feat] = feature_list
        data_train_test_concat.append((pd.DataFrame(features_concat), target_concat))
    
    
    return (data_train_test_concat[0][0], data_train_test_concat[1][0], 
            data_train_test_concat[0][1], data_train_test_concat[1][1])

if __name__ == '__main__':
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/met/'
    features_train = pd.read_pickle(path + 'features_train_met.p')
    target_train = pd.read_pickle(path + 'target_train_met.p')
    features_test = pd.read_pickle(path + 'features_test_met.p')
    target_test = pd.read_pickle(path + 'target_test_met.p')
    
    features_train_concat, target_train_concat, features_test_concat, target_test_concat = concatenate(features_train,
                                                                                                       features_test,
                                                                                                       target_train,
                                                                                                       target_test)
    pd.to_pickle(features_train_concat, path + 'features_train_concat.p')
    pd.to_pickle(features_train_concat, path + 'target_train_concat.p')
    pd.to_pickle(features_train_concat, path + 'features_test_concat.p')
    pd.to_pickle(features_train_concat, path + 'target_test_concat.p')