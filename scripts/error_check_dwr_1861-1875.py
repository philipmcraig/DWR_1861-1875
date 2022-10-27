# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:01:03 2021

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
import os

homedir = 'C:/Users/phili/Github/DWR_1861-1875/'
#'C:/Users/qx911590/Documents/dwr-1861-1875/'

#year = '1869'
years = pl.linspace(1861,1875,15).astype(int)

tot_err = 0
tot_emp = 0
tot_nan = 0
tot_gd = 0
pres_count = 0
rain_count = 0
tdry_count = 0
twet_count = 0

for Y in range(len(years)):
    filenames = glob.glob(homedir+'finished/DWR_'+str(years[Y])+'*')
    
    for i in range(len(filenames)):
        df = pd.read_csv(filenames[i],header=None,skiprows=1)#,names=names)
        logs = pl.array(df)
        errs = pl.where((logs==-999) | (logs==' -999.00') | (logs==' -999') | (logs=='Inf') | (logs==' Inf'))
        empty = pl.where((logs==999) | (logs==' 999.00'))
        miss = pl.where(logs==' NaN')
        good = pl.where((logs!=999) | (logs!=' 999.00') | (logs!=' NaN'))
        if years[Y] < 1872:
            pres_gd = pl.where((logs[:,1]!=999) & (logs[:,1]!=' 999.00') & (logs[:,1]!=' NaN'))
            PG = pres_gd[0].size
            rain_gd = pl.where((logs[:,-1]!=999) & (logs[:,-1]!=' 999.00') & (logs[:,-1]!=' NaN'))
            tdry_gd = pl.where((logs[:,2]!=999) & (logs[:,2]!=' 999.00') & (logs[:,2]!=' NaN'))
            TDG = tdry_gd[0].size
        else:
            pg1 = pl.where((logs[:,1]!=999) & (logs[:,1]!=' 999.00') & (logs[:,1]!=' NaN'))
            pg2 = pl.where((logs[:,-3]!=999) & (logs[:,-3]!=' 999.00') & (logs[:,-3]!=' NaN'))
            PG = pg1[0].size + pg2[0].size
            rain_gd = pl.where((logs[:,-4]!=999) & (logs[:,-4]!=' 999.00') & (logs[:,-4]!=' NaN'))
            tdg1 = pl.where((logs[:,2]!=999) & (logs[:,2]!=' 999.00') & (logs[:,2]!=' NaN'))
            tdg2 = pl.where((logs[:,-2]!=999) & (logs[:,-2]!=' 999.00') & (logs[:,-2]!=' NaN'))
            TDG = tdg1[0].size + tdg2[0].size
        
        if years[Y] < 1864:
            twet_gd = pl.where((logs[:,3]!=999) & (logs[:,3]!='999') & \
                                (logs[:,3]!=' 999.00') & (logs[:,3]!=' NaN'))
            TWG = twet_gd[0].size
        elif years[Y] >= 1872:
            twg1 = pl.where((logs[:,3]!=999) & (logs[:,3]!='999') & \
                                (logs[:,3]!=' 999.00') & (logs[:,3]!=' NaN'))
            twg2 = pl.where((logs[:,-1]!=999) & (logs[:,-1]!='999') & \
                                (logs[:,-1]!=' 999.00') & (logs[:,-1]!=' NaN'))
            TWG = twg1[0].size + twg2[0].size
        #errs = pl.where((logs[:,1:].astype(float)==-999) | (logs[:,1:].astype(float)==pl.float32('inf')))
    #    empty = pl.where(logs[:,1:].astype(float)==999)
        #miss = pl.where(logs[:,1:].astype(float)==pl.float32('nan'))
    #    
        tot_err = tot_err + errs[0].size
        tot_emp = tot_emp + empty[0].size
        tot_nan = tot_nan + miss[0].size
        tot_gd = tot_gd + good[0].size
        pres_count = pres_count + PG#pres_gd[0].size
        rain_count = rain_count + rain_gd[0].size
        tdry_count = tdry_count + TDG
        twet_count = twet_count + TWG
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

#print 'Total number of errors = ', tot_err
#print 'Total number of empty values = ', tot_emp
#print 'Total number of missing values = ', tot_nan
#print 'Total number of good values = ', tot_gd
print 'Total number of pressure values = ', pres_count
print 'Total number of rain values = ', rain_count
print 'Total number of tdry values = ', tdry_count
print 'Total number of twet values = ', twet_count