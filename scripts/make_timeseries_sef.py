# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:08:55 2019

@author: pmcraig
"""

import pylab as pl
import pandas as pd
import glob

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
dwrdir = ncasdir + 'DWRcsv/'
sefdir = ncasdir + 'Data_for_SEF/'

years = pl.linspace(1900,1910,11).astype(int)
loc = 'Aberdeen'
rainlist = []
datelist = []
yearlist = []
monthlist = []
daylist = []
count = 0

for Y in range(years.size):
    print years[Y]
    dwrfiles = glob.glob(dwrdir+str(years[Y])+'/*')

    rainyr = pl.zeros([len(dwrfiles)])
    dates = pl.zeros([len(dwrfiles)],dtype='S10')
    for i in range(len(dwrfiles)):
        df = pd.read_csv(dwrfiles[i],header=None)
        logs = pl.array(df)
        ind = pl.where(logs[:,0]==loc+' '); ind = ind[0][0]
        
        if logs[ind,-1] == ' ' or logs[ind,-1] == '  ':
            rainyr[i] = pl.float32('nan')
        else:
            rainyr[i] = pl.float32(logs[ind,-1])*25.4
        
        
        yr = dwrfiles[i][62:66]
        mon = dwrfiles[i][67:69]
        day = dwrfiles[i][70:72]
        dates[i] = yr+'/'+mon+'/'+day
        yearlist.append(int(yr))
        monthlist.append(int(mon))
        daylist.append(int(day))
        
        count = count + 1
        if yr == '1908' and mon == '07' and day == '01':
            A = count - 1
        
    rainlist.append(rainyr)
    datelist.append(dates)

rainseries = pl.concatenate(rainlist).ravel(); rainseries = list(rainseries)
dateseries = pl.concatenate(datelist).ravel(); dateseries = list(dateseries)

X = pl.argwhere(pl.isnan(rainseries))

for i in range(len(X)):
    rainseries.pop(X[i,0]-i)
    dateseries.pop(X[i,0]-i)
    yearlist.pop(X[i,0]-i)
    monthlist.pop(X[i,0]-i)
    daylist.pop(X[i,0]-i)
    
rainseries = pl.asarray(rainseries)
dateseries = pl.asarray(dateseries)
yearseries = pl.asarray(yearlist)
monthseries = pl.asarray(monthlist)
dayseries = pl.asarray(daylist)

hours = pl.zeros_like(dayseries); mins = pl.zeros_like(dayseries)
hours[:A-len(X)] = 8; hours[A-len(X):] = 7

data = pl.array([yearseries,monthseries,dayseries,hours,mins,rainseries])
data = pd.DataFrame(data.T)

data.to_csv(sefdir+loc+'_rainfall.csv',index=False,
            header=['year','month','day','hour','minute','rainfall'])
#pl.savetxt(sefdir+loc+'_rainfall.csv',data.T,delimiter=',')