# -*- coding: utf-8 -*-
import pandas as pd

def remove_outliers(df_soil):
    
    #remove land where soil data has label 7: water bodies, 6: permafrost and 5: Mainly non-soil
    df_soil = df_soil[~(df_soil['SQ1'] >= 5)]
    
    #remove land covered by more than 50% water
    df_soil = df_soil[~(df_soil['WAT_LAND'] >= 50)]
    
    
    return df_soil


