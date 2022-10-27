# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:08:55 2019

@author: pmcraig
"""

import pylab as pl
import pandas as pd
import glob
#import SEF

def CombineSeries(list1,list2):
    """
    """
    combined = pl.zeros([len(list1)+len(list2)])
    combined[0::2] = list1[:]
    combined[1::2] = list2[:]
    combined = list(combined)
    
    return combined

exec(open('/home/users/pmcraig/create.py').read())
exec(open('/home/users/pmcraig/write.py').read())

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
dwrdir = ncasdir + 'DWRcsv/'
sefdir = ncasdir + 'Data_for_SEF/'

years = pl.linspace(1900,1910,11).astype(int)
loc = 'Aberdeen'
preslist_eve = []; preslist_mor = []
#datelist_eve = []; datelist_mor = []
yearlist_eve = [1899]; yearlist_mor = []
monthlist_eve = [12]; monthlist_mor = []
daylist_eve = [31]; daylist_mor = []
count = 0

for Y in range(years.size):
    print(years[Y])
    dwrfiles = glob.glob(dwrdir+str(years[Y])+'/*')

    evepres_yr = pl.zeros([len(dwrfiles)])
    morpres_yr = pl.zeros([len(dwrfiles)])
    dates = pl.zeros([len(dwrfiles)],dtype='S10')
    for i in range(len(dwrfiles)):
        df = pd.read_csv(dwrfiles[i],header=None)
        logs = pl.array(df)
        ind = pl.where(logs[:,0]==loc+' '); ind = ind[0][0]
        
        if logs[ind,1] == ' ' or logs[ind,1] == '  ':
            evepres_yr[i] = pl.float32('nan')
        else:
            evepres_yr[i] = pl.float32(logs[ind,1])*33.8639
        
        if logs[ind,3] == ' ' or logs[ind,3] == '  ':
            morpres_yr[i] = pl.float32('nan')
        else:
            morpres_yr[i] = pl.float32(logs[ind,3])*33.8639
        
        
        yr = dwrfiles[i][62:66]
        mon = dwrfiles[i][67:69]
        day = dwrfiles[i][70:72]
        dates[i] = yr+'/'+mon+'/'+day
        yearlist_eve.append(int(yr)); yearlist_mor.append(int(yr))
        monthlist_eve.append(int(mon)); monthlist_mor.append(int(mon))
        daylist_eve.append(int(day)); daylist_mor.append(int(day))
        
        count = count + 1
        if yr == '1908' and mon == '07' and day == '01':
            A = count - 1
        
    preslist_eve.append(evepres_yr)
    preslist_mor.append(morpres_yr)
    #datelist_eve.append(dates); datelist_mor.append(dates)


yearlist_eve.pop(-1)
monthlist_eve.pop(-1)
daylist_eve.pop(-1)

presseries_eve = pl.concatenate(preslist_eve).ravel()
presseries_mor = pl.concatenate(preslist_mor).ravel()

presseries_eve = list(presseries_eve)
presseries_mor = list(presseries_mor)




hours_eve = pl.zeros_like(daylist_eve); hours_eve[:] = 18
hours_mor = pl.zeros_like(daylist_eve)
hours_mor[:A] = 8; hours_mor[A:] = 7

hours_eve = list(hours_eve); hours_mor = list(hours_mor)

presseries_all = CombineSeries(presseries_eve,presseries_mor)
yearseries_all = CombineSeries(yearlist_eve,yearlist_mor)
monthseries_all = CombineSeries(monthlist_eve,monthlist_mor)
dayseries_all = CombineSeries(daylist_eve,daylist_mor)
hours = CombineSeries(hours_eve,hours_mor)

X = pl.argwhere(pl.isnan(presseries_all))
#X_eve = pl.argwhere(pl.isnan(presseries_eve))
#X_mor = pl.argwhere(pl.isnan(presseries_mor))

for i in range(len(X)):
    presseries_all.pop(X[i,0]-i)
#    dateseries.pop(X[i,0]-i)
    yearseries_all.pop(X[i,0]-i)
    monthseries_all.pop(X[i,0]-i)
    dayseries_all.pop(X[i,0]-i)
    hours.pop(X[i,0]-i)

#for i in range(len(X_eve)):
#    presseries_eve.pop(X_eve[i,0]-i)
#    yearlist_eve.pop(X_eve[i,0]-i)
#    monthlist_eve.pop(X_eve[i,0]-i)
#    daylist_eve.pop(X_eve[i,0]-i)
#    hours_eve.pop(X_eve[i,0]-i)
#
#for i in range(len(X_mor)):
#    presseries_mor.pop(X_mor[i,0]-i)
#    yearlist_mor.pop(X_mor[i,0]-i)
#    monthlist_mor.pop(X_mor[i,0]-i)
#    daylist_mor.pop(X_mor[i,0]-i)
#    hours_mor.pop(X_eve[i,0]-i)


presseries_all = pl.asarray(presseries_all)
#dateseries = pl.asarray(dateseries)
yearseries_all = pl.asarray(yearseries_all)
monthseries_all = pl.asarray(monthseries_all)
dayseries_all = pl.asarray(dayseries_all)
hours = pl.asarray(hours)

#hours = pl.zeros_like(dayseries_all)
mins = pl.zeros_like(dayseries_all)
#hours[:2*A-len(X)] = 8; hours[2*A-len(X):] = 7
#
data = pl.array([yearseries_all,monthseries_all,dayseries_all,hours,mins,
                                     presseries_all])
#data = pd.DataFrame(data.T)

#data.to_csv(sefdir+loc+'_mslp.csv',index=False,
#            header=['year','month','day','hour','minute','mslp'])
#pl.savetxt(sefdir+loc+'_mslp.csv',data.T,delimiter=',')

#perlist = pl.zeros(8026)
records = {
           'ID': 'DWR_Aberdeen', 'Name': 'Aberdeen Observatory', 'Source': 'C3S_Aberdeen', 
           'Lat': -34.0958, 'Lon': -59.0242,  
           'Vbl': 'mslp', 'Stat': 'point', 'Units': 'hPa', 
           'Year': data[0,:], 
           'Month': data[1,:], 'Day': data[2,:],
           'Hour': data[3,:], 'Minute': data[4,:], 
           'Period': 0,
           'Value': data[-1,:]
           }
#           'Meta2': ['Orig=766.9mm|Orig.time=', 'Orig.=764.7mm|Orig.time='], 
#           'orig_time': ['7am', '7am']

obs = create(records)
write_file(obs, sefdir+'test.tsv')
