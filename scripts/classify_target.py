#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 13:45:24 2025

@author: jane
"""

import pandas as pd




def classify(target_train, target_test):
    '''
    Classify Drought Targets as 0 if drought levels are 0 or 1 and as 1 if 
    drought levels are higher. Write results to pickle
    Parameters
    ----------
    target_train : 1D pandas DataFrame
        Target of Training Data.
    target_test : 1D pandas DataFrame
        Target of Testing Data.

    Returns
    -------
    None.
    '''

    Parameters
    ----------
    target_train : TYPE
        DESCRIPTION.
    target_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    target_train_class = target_train >= 2
    target_test_class = target_test >= 2
    '''
    pd.to_pickle(target_train_class, path + 'target_met_train_class.p')
    pd.to_pickle(target_test_class, path + 'target_met_test_class.p')
    '''
    return

if __name__ == '__main__':
    
    path = '/home/jane/Documents/Weiterbildung/DPP/portfolio_project/data/processed/met/'
    target_train = pd.read_pickle(path + 'target_met_train_prepared.p')
    target_test = pd.read_pickle(path + 'target_met_test_prepared.p')
    
    classify(target_train, target_test)