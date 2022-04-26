# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pylab as pl
import pandas as pd
import glob
import os

homedir = 'C:/Users/qx911590/Documents/dwr-1861-1875/'

filenames = glob.glob(homedir+'allfiles/*')

tot_err = 0; te = pl.zeros([len(filenames)])
tot_nan = 0; tn = te.copy()

for i in range(len(filenames)):
    num_lines = sum(1 for line in open(filenames[i]))
    if num_lines <= 1:
        te[i] = tot_err
        tn[i] = tot_nan
        continue
    
    df = pd.read_csv(filenames[i],header=None,skiprows=1)#,names=names)
    logs = pl.array(df)
    errs = pl.where(logs==-999)
    miss = pl.where(logs==' NaN')
    
    tot_err = tot_err + errs[0].size; te[i] = tot_err
    tot_nan = tot_nan + miss[0].size; tn[i] = tot_nan
    
    #if i == 0:
    nrow = logs.shape[0]
    ncol = logs.shape[1]
    
    if i > 0:
        if nrow != prevrow or ncol != prevcol:
            print(i, filenames[i])
        
    prevcol = ncol
    prevrow = nrow
        




print('Total number of errors = ', tot_err)
print('Total number of missing values = ', tot_nan)