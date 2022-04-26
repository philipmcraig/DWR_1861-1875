# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:01:03 2021

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
import os

homedir = 'C:/Users/qx911590/Documents/dwr-1861-1875/'

#year = '1869'
years = pl.linspace(1861,1875,15).astype(int)

tot_err = 0
tot_emp = 0
tot_nan = 0
tot_gd = 0

for Y in range(len(years)):
    filenames = glob.glob(homedir+'finished/DWR_'+str(years[Y])+'*')
    
    for i in range(len(filenames)):
        df = pd.read_csv(filenames[i],header=None,skiprows=1)#,names=names)
        logs = pl.array(df)
        errs = pl.where((logs==-999) | (logs==' -999.00') | (logs==' -999') | (logs=='Inf') | (logs==' Inf'))
        empty = pl.where((logs==999) | (logs==' 999.00'))
        miss = pl.where(logs==' NaN')
        good = pl.where((logs!=999) | (logs!=' 999.00') | (logs!=' NaN'))
        #errs = pl.where((logs[:,1:].astype(float)==-999) | (logs[:,1:].astype(float)==pl.float32('inf')))
    #    empty = pl.where(logs[:,1:].astype(float)==999)
        #miss = pl.where(logs[:,1:].astype(float)==pl.float32('nan'))
    #    
        tot_err = tot_err + errs[0].size
        tot_emp = tot_emp + empty[0].size
        tot_nan = tot_nan + miss[0].size
        tot_gd = tot_gd + good[0].size
 
    #    
    #    if errs[0].size > 0:
    #        print filenames[i]
        
        #GRT = pl.where((logs[:,5].astype(float)>logs[:,4].astype(float)) & (logs[:,5].astype(float)!=999))
        
    #    if GRT[0].size > 0:
    #        print filenames[i], logs[:,0][GRT[0]]
        
    #    PLIM = pl.where((logs[:,1].astype(float)>31.0) & (logs[:,1].astype(float)<999.0))
    #    
    #    if PLIM[0].size > 0:
    #        print filenames[i], logs[:,0][PLIM[0]]

print 'Total number of errors = ', tot_err
print 'Total number of empty values = ', tot_emp
print 'Total number of missing values = ', tot_nan
print 'Total number of good values = ', tot_gd