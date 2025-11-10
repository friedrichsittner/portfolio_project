#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 15:55:04 2025

@author: jane
"""

import pandas as pd
import numpy as np

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
    datasets = {'train' : (features_train, target_train),
                'test' : (features_test, target_test)}
    
    for name,  (features, target) in datasets.items():
        df = features.copy()
        df['score'] = target
        df_grouped = df.groupby('fips')
        feature_concat_list = []
        target_concat_list = []
        for group in df_grouped:
            fips_feat, fips_target = group[1].drop('score', axis = 1), group[1].score
            fips_feat = fips_feat.reset_index()
            fips_target = fips_target.reset_index()
            target_concat = fips_target.dropna()
            cut_points = target_concat.index.to_numpy()
            seg_ids = np.searchsorted(cut_points, np.arange(len(fips_feat)), side = 'right')
            fips_feat_grouped = fips_feat.groupby(seg_ids).mean()[:len(target_concat)]
            feature_concat_list.append(fips_feat_grouped)
            target_concat_list.append(target_concat)
        feature_concat_df = pd.concat(feature_concat_list).reset_index().drop(['level_0', 'index'], axis = 1)
        target_concat_df = pd.concat(target_concat_list).reset_index().drop(['level_0', 'index'], axis = 1)
        data_train_test_concat.append((feature_concat_df, target_concat_df))
        print('Unique Fips for {}set: '.format(name))
    return (data_train_test_concat[0][0], data_train_test_concat[0][1], 
            data_train_test_concat[1][0], data_train_test_concat[1][1])

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
    pd.to_pickle(target_train_concat, path + 'target_train_concat.p')
    pd.to_pickle(features_test_concat, path + 'features_test_concat.p')
    pd.to_pickle(target_test_concat, path + 'target_test_concat.p')