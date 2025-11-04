# -*- coding: utf-8 -*-
import pandas as pd

def remove_outliers(features_soil, target_soil):
    
    #remove land where soil data has label 7: water bodies, 6: permafrost and 5: Mainly non-soil
    mask_SQ = ~(features_soil['SQ1'] >= 5)
    features_soil = features_soil[mask_SQ]
    print(target_soil.index)
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

if __name__ == '__main__':
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/soil/'
    features_train = pd.read_pickle(path + 'features_train_soil.p')
    target_train = pd.read_pickle(path + 'target_train_soil.p')
    
    remove_outliers(features_train, target_train)