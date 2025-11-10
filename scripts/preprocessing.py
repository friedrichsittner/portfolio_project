# -*- coding: utf-8 -*-
import pandas as pd

def remove_outliers(features_soil, features_met, target_soil, target_met):
    '''
    Removes Outliers from Soil Data and corresponding lines from meteorological data

    Parameters
    ----------
    features_soil : pandas DataFrame
        Training Features for Soil Data
    features_met : pandas DataFrame
        Training Features for Meteorological Data
    target_soil : pandas DataFrame
        Training Target for Soil Data
    target_met : pandas DataFrame
        Training Target for Meteorological Data

    Returns
    -------
    features_soil : pandas DataFrame
        Training Features for Soil Data
    features_met : pandas DataFrame
        Training Features for Meteorological Data
    target_soil : pandas DataFrame
        Training Target for Soil Data
    target_met : pandas DataFrame
        Training Target for Meteorological Data

    '''
    #remove land where soil data has label 7: water bodies, 6: permafrost and 5: Mainly non-soil
    mask_SQ = ~(features_soil['SQ1'] >= 5)
    features_soil = features_soil[mask_SQ]
    target_soil = target_soil[mask_SQ]
    mask_met = pd.Series([True]*len(features_met), index = features_met.index)
    for fips in mask_SQ[mask_SQ == False].index:
        mask_met &= ~(features_met['fips'] == fips)

    #remove land covered by more than 50% water
    mask_WAT_LAND = ~(features_soil['WAT_LAND'] >= 50)
    features_soil = features_soil[mask_WAT_LAND]
    target_soil = target_soil[mask_WAT_LAND]
    for fips in mask_WAT_LAND[mask_WAT_LAND == False].index:
        mask_met &= ~(features_met['fips'] == fips)

    #remove rows where score is not given
    mask_score = ~target_soil.isna()
    features_soil = features_soil[mask_score[0]]
    target_soil = target_soil[mask_score[0]]
    for fips in mask_score[mask_score[0] == False].index:
        mask_met &= ~(features_met['fips'] == fips)
    
    
    features_met = features_met.drop(mask_met[mask_met == False].index)
    target_met = target_met.drop(mask_met[mask_met == False].index)
    return features_soil, features_met, target_soil, target_met

if __name__ == '__main__':
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/'
    features_soil = pd.read_pickle(path + 'soil/features_train_soil.p')
    target_soil = pd.read_pickle(path + 'soil/target_train_soil.p')
    features_met = pd.read_pickle(path + 'met/features_train_met.p')
    target_met = pd.read_pickle(path + 'met/target_train_met.p')
    
    remove_outliers(features_soil, features_met, target_soil, target_met)