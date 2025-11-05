# -*- coding: utf-8 -*-
import pandas as pd

def remove_outliers(features_soil, features_met, target_soil, target_met):
    
    #remove land where soil data has label 7: water bodies, 6: permafrost and 5: Mainly non-soil
    mask_SQ = ~(features_soil['SQ1'] >= 5)
    features_soil = features_soil[mask_SQ]
    target_soil = target_soil[mask_SQ]
    mask_SQ_met = pd.Series([True]*len(features_met), index = features_met.index)
    print(mask_SQ_met.shape)
    print((features_met['fips'] == 1001).shape)
    for fips in mask_SQ[mask_SQ == False].index:
        mask_SQ_met &= (features_met['fips'] == fips)
    
    features_met = features_met.drop(mask_SQ[mask_SQ_met == False].index)
    print(len(features_met))
    
    #remove land covered by more than 50% water
    mask_WAT_LAND = ~(features_soil['WAT_LAND'] >= 50)
    features_soil = features_soil[mask_WAT_LAND]
    target_soil = target_soil[mask_WAT_LAND]
    
    #remove rows where score is not given
    mask_score = ~target_soil.isna()
    features_soil = features_soil[mask_score[0]]
    target_soil = target_soil[mask_score[0]]
    
    return features_soil, target_soil

if __name__ == '__main__':
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/'
    features_soil = pd.read_pickle(path + 'soil/features_train_soil.p')
    target_soil = pd.read_pickle(path + 'soil/target_train_soil.p')
    features_met = pd.read_pickle(path + 'met/features_train_met.p')
    target_met = pd.read_pickle(path + 'met/target_train_met.p')
    
    remove_outliers(features_soil, features_met, target_soil, target_met)