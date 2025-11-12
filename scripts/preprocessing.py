# -*- coding: utf-8 -*-
import pandas as pd

def remove_outliers_soil(features_soil, target_soil):
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

    #remove land covered by more than 50% water
    mask_WAT_LAND = ~(features_soil['WAT_LAND'] >= 50)
    features_soil = features_soil[mask_WAT_LAND]
    target_soil = target_soil[mask_WAT_LAND]

    #remove rows where score is not given
    mask_score = ~target_soil.isna()
    features_soil = features_soil[mask_score[0]]
    target_soil = target_soil[mask_score[0]]

    return features_soil, target_soil

def remove_soil_fips_from_met(features_soil, features_met, target_met):
    '''
    Remove features correspond to fips removed in soil data

    Parameters
    ----------
    features_soil : pd.Dataframe
        soil features that have had their outliers removed
    features_met : pd.Dataframe
    target_met : 1D pd.Dataframe

    Returns
    -------
    features_met : pd.Dataframe
    target_met : 1D pd.Dataframe

    '''
    mask = features_met.fips.isin(features_soil.index)
    features_met = features_met[mask]
    target_met = target_met[mask]
    return features_met, target_met

if __name__ == '__main__':
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/'
    features_soil = pd.read_pickle(path + 'soil/features_train_soil.p')
    target_soil = pd.read_pickle(path + 'soil/target_train_soil.p')
    
    remove_outliers_soil(features_soil, target_soil)