#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:34:38 2025

@author: jane
"""
import pandas as pd
import pickle
from preprocessing import remove_outliers

def feature_preperation(features_train_soil, features_test_soil, 
                        target_train_soil, target_test_soil,
                        features_train_met, features_test_met,
                        target_train_met, target_test_met,
                        model_soil):

    #clean data and make soil prediction------------------------------------------
    
    features_soil = pd.concat([features_train_soil, features_test_soil])
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
    
    pd.to_pickle(features_train_met, path_met + 'features_met_train_prepared.p')
    pd.to_pickle(features_test_met, path_met + 'features_met_test_prepared.p')
    pd.to_pickle(target_train_met, path_met + 'target_met_train_prepared.p')
    pd.to_pickle(target_test_met, path_met + 'target_met_test_prepared.p')
    
    #split by region and save seperately----------------------------------------
    regions = {'west' : [4, 6, 8, 16, 30, 32, 35, 41, 49, 53, 56],
               'south' : [1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54],
               'midwest' : [17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55],
               'east' : [9, 23, 25, 33, 34, 36, 42, 44, 50]}
    for region in regions.keys():
        region_fips_list = []
        for state in regions[region]:
            region_fips_list.extend(features_soil[(features_soil.index >= 1000*state) & (features_soil.index < 1000*(state + 1))].index)
        features_train_region = features_train_met[features_train_met.fips.isin(region_fips_list)]
        features_test_region = features_test_met[features_test_met.fips.isin(region_fips_list)]
        target_train_region = target_train_met[features_train_met.fips.isin(region_fips_list)]
        target_test_region = target_test_met[features_test_met.fips.isin(region_fips_list)]
        pd.to_pickle(features_train_region, 
                     path_met + 'features_train_met_prep_{}.p'.format(region))
        pd.to_pickle(features_test_region, 
                     path_met + 'features_test_met_prep_{}.p'.format(region))
        pd.to_pickle(target_train_region, 
                     path_met + 'target_train_met_prep_{}.p'.format(region))
        pd.to_pickle(target_test_region, 
                     path_met + 'target_test_met_prep_{}.p'.format(region))
        
if __name__ == '__main__':
    path_soil = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/soil/'
    features_train_soil = pd.read_pickle(path_soil + 'features_train_soil.p')
    target_train_soil = pd.read_pickle(path_soil + 'target_train_soil.p')
    features_test_soil = pd.read_pickle(path_soil + 'features_test_soil.p')
    target_test_soil = pd.read_pickle(path_soil + 'target_test_soil.p')
    
    path_met = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/met/'
    features_train_met = pd.read_pickle(path_met + 'features_train_concat.p')
    target_train_met = pd.read_pickle(path_met + 'target_train_concat.p')
    features_test_met = pd.read_pickle(path_met + 'features_test_concat.p')
    target_test_met = pd.read_pickle(path_met + 'target_test_concat.p')
    
    model_soil = pickle.load(open('/home/jane/Documents/Weiterbildung/DPP/portfolio_project/scripts/trained_model_soil.p', 'rb'))
    
    feature_preperation(features_train_soil, features_test_soil, 
                            target_train_soil, target_test_soil,
                            features_train_met, features_test_met,
                            target_train_met, target_test_met,
                            model_soil)