# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:14:56 2022

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
import os

homedir = 'C:/Users/qx911590/Documents/dwr-1861-1875/'

years = pl.linspace(1861,1875,15).astype(int)

statnames = []

for i in range(len(years)):
    filenames = glob.glob(homedir+'finished/DWR_'+str(years[i])+'*')
    
    for name in range(len(filenames)):
        df = pd.read_csv(filenames[name],header=None,skiprows=1)#,names=names)
        logs = pl.array(df)
        
        stations = logs[:,0]
        
        for s in range(len(stations)):
            if stations[s].replace(' ','') in statnames:
                pass
            else:
                statnames.append(stations[s].replace(' ',''))
                if len(statnames) == 36:
                    print filenames[name]