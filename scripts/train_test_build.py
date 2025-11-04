# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    '''
    Create Training, Test and Validation Datasets for meteorological and soil data.
    Meteorological Training Data covers 2017-2018, Testing is 2019 and Validation 2020.
    Soil Data is split randomly between train and test data, no validation data.
    '''
    
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/'
    
    df_soil = pd.read_csv(path + 'raw/us-drought-meteorological-data/soil_data.csv', index_col = 'fips')
    #df_soil = df_soil.drop('fips', axis = 1)
    df_met_train = pd.read_csv(path + 'raw/met_train.csv')
    df_met_test_val = pd.read_csv(path + 'raw/met_test_val.csv')
    
    df_met_train.date = pd.to_datetime(df_met_train.date)
    df_met_test_val.date = pd.to_datetime(df_met_test_val.date)
    
    df_met_test = df_met_test_val[df_met_test_val.date.dt.year == 2019]
    df_met_val = df_met_test_val[df_met_test_val.date.dt.year == 2020]
    
    features_train_met = df_met_train.drop('score', axis = 1)
    features_test_met = df_met_test.drop('score', axis = 1)
    features_val_met = df_met_test.drop('score', axis = 1)
    
    target_train_met = df_met_train['score']
    target_test_met = df_met_test['score']
    
    av_drought_list = []
    for fips in df_soil.index:
        df_met_train_fips = df_met_train[df_met_train.fips == fips]
        df_met_test_fips = df_met_test[df_met_test.fips == fips]
        av_met_train_fips = df_met_train_fips.score.mean()
        av_met_test_fips = df_met_test_fips.score.mean()
        av_drought_list.append((av_met_train_fips + av_met_test_fips)/2)
    
    target_soil = pd.DataFrame(av_drought_list, index = df_soil.index)
    
    features_train_soil, features_test_soil, target_train_soil, target_test_soil = train_test_split(df_soil,
                                                                                                    target_soil,
                                                                                                    test_size = 0.25,
                                                                                                    random_state = 37)
    
    pd.to_pickle(features_train_soil, path + 'processed/soil/features_train_soil.p')
    pd.to_pickle(features_test_soil, path + 'processed/soil/features_test_soil.p')
    pd.to_pickle(target_train_soil, path + 'processed/soil/target_train_soil.p')
    pd.to_pickle(target_test_soil, path + 'processed/soil/target_test_soil.p')
    pd.to_pickle(features_train_met, path + 'processed/met/features_train_met.p')
    pd.to_pickle(features_test_met, path + 'processed/met/features_test_met.p')
    pd.to_pickle(features_val_met, path + 'processed/met/features_val_met.p')
    pd.to_pickle(target_train_met, path + 'processed/met/target_train_met.p')
    pd.to_pickle(target_test_met, path + 'processed/met/target_test_met.p')