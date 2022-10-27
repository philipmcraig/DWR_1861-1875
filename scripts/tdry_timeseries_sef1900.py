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
    import pylab as pl
    combined = pl.zeros([len(list1)+len(list2)],dtype='object')
    combined[0::2] = list1[:]
    combined[1::2] = list2[:]
    combined = list(combined)
    
    return combined

def Get2pmData(dwrdir,locstring,varstring,):
    """
    """
    import pylab as pl
    import pandas as pd
    
    df = pd.read_csv(dwrdir+'1900_2pm.csv',header=0)
    S = pl.where(df.columns.values==locstring+' '+varstring)
    if S[0].size == 1:
        print "##### 2pm data for " + locstring + " tdry exists #####"
        data2pm = pl.array(df)
        tdry2pm = (pl.array(df[locstring+' '+varstring])[:-1]-32)*(5./9.)
        orig2pm = pl.array(df[locstring+' '+varstring])[:-1]
        years2pm = pl.array([x[-4:] for x in data2pm[:-1,0]])
        months2pm = pl.array([x[3:5] for x in data2pm[:-1,0]])
        days2pm = pl.array([x[:2] for x in data2pm[:-1,0]])
        #hours2pm = pl.zeros_like(pres2pm); hours2pm[:] = 14
        meta2pm = pl.zeros_like(tdry2pm,dtype='object')
        for j in range(len(meta2pm)):
            if pl.isnan(orig2pm[j])==True:
                meta2pm[j] = 'nan'
            else:
                meta2pm[j] = 'orig='+str(round(orig2pm[j],2))+'inHg'
        
        return tdry2pm, meta2pm, years2pm, months2pm, days2pm#, hours2pm

def MetaData(splitpoint,origseries_eve,origseries_mor):
    """
    """
    import pylab as pl
    print "##### processing tdry metadata #####"
    meta_eve = pl.zeros([len(origseries_eve)],dtype='object')
    meta_mor = pl.zeros([len(origseries_mor)],dtype='object')
#    if splitpoint == None:
#        meta_eve[:] = 'PGC=N'; meta_mor[:] = 'PGC=N'
#    else:
#    meta_eve[:splitpoint] = 'PGC=N|'; meta_eve[splitpoint:] = 'PGC=Y|'
#    meta_mor[:splitpoint] = 'PGC=N|'; meta_mor[splitpoint:] = 'PGC=Y|'
    for i in range(meta_eve.size):
        if pl.isnan(origseries_eve[i])== True:
            pass
        else:
            meta_eve[i] = 'orig=' + str(round(origseries_eve[i],2)) +'inHg'
        
        if pl.isnan(origseries_mor[i])==True:
            pass
        else:
            meta_mor[i] = 'orig=' + str(round(origseries_mor[i],2)) +'inHg'
    meta_eve = list(meta_eve); meta_mor = list(meta_mor)
    print "##### tdry metadata completed #####"
    
    return meta_eve, meta_mor

def Include2pm(current,data2pm):
    """
    """
    import pylab as pl
    current = pl.array(current)
    alldata = pl.zeros([current.size + data2pm.size],dtype='object')
    alldata[0:365*3:3] = data2pm
    alldata[1:365*3:3] = current[0:365*2:2]
    alldata[2:365*3:3] = current[1:365*2:2]
    alldata[365*3:] = current[365*2:]
    alldata = list(alldata)
    
    return alldata

#exec(open('/home/users/pmcraig/create.py').read())
#exec(open('/home/users/pmcraig/SEF-Python-master/build/lib/SEF/create.py').read())

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
dwrdir = ncasdir + 'DWRcsv/'
sefdir = ncasdir + 'Data_for_SEF/'

years = pl.array([1900]).astype(int)#pl.linspace(1900,1904,5).astype(int)
exec(open('/home/users/pmcraig/statname.py').read())#loc = 'Aberdeen'


print "##### running " + loc + " tdry #####"

tdry2pm, meta2pm, years2pm, months2pm, days2pm = Get2pmData(dwrdir,loc,'Dry')

tdry_yr = pl.zeros([len(tdry2pm)])
orig_yr = pl.zeros([len(tdry2pm)])

Z = pl.where(pl.array(tdry2pm)!=0)
tdryseries = list(pl.array(tdry2pm)[Z[0]])
yearseries = list(pl.array(years2pm)[Z[0]])
monthseries = list(pl.array(months2pm)[Z[0]])
dayseries = list(pl.array(days2pm)[Z[0]])
#hours = list(pl.array(hours)[Z[0]])
#mins = list(pl.array(mins)[Z[0]])
meta = list(pl.array(meta2pm)[Z[0]])

X = pl.argwhere(pl.isnan(tdryseries))

for i in range(len(X)):
    tdryseries.pop(X[i,0]-i)
    yearseries.pop(X[i,0]-i)
    monthseries.pop(X[i,0]-i)
    dayseries.pop(X[i,0]-i)
    meta.pop(X[i,0]-i)


tdryseries = pl.asarray(tdryseries[:])
yearseries = pl.asarray(yearseries[:])
monthseries = pl.asarray(monthseries[:])
dayseries = pl.asarray(dayseries[:])
#hours = pl.asarray(hours[:])
#mins = pl.asarray(mins)
meta = pl.asarray(meta[:])

hours = pl.zeros_like(dayseries); hours[:] = 14
mins = pl.zeros([dayseries.size])#; mins[:] = 43
#hours[:2*A-len(X)] = 8; hours[2*A-len(X):] = 7
#
data = pl.array([yearseries,monthseries.astype(int),
                 dayseries.astype(int),hours,mins,
                    pl.around(tdryseries,2),meta],dtype='object')
data = pd.DataFrame(data.T)

data.to_csv(sefdir+loc.replace(' ','')+'_tdry2pm.csv',index=False,
            header=['year','month','day','hour','minute','tdry','meta'])

print "##### completed " + loc + " tdry #####"

#perlist = pl.zeros(8026)#; perlist = list(perlist)
#m2 = pl.zeros_like(data[1,:],dtype='S4'); m2[:] = 'none'#; m2  = list(m2)
#ot = pl.zeros_like(data[1,:],dtype='S1'); ot[:] = '?'#; ot = list(ot)
#records = {
#           'ID': 'DWR_Aberdeen', 'Name': 'AberdeenObservatory', 'Source': 'C3S_Aberdeen', 
#           'Lat': -2.100822, 'Lon': 57.164128, 'Alt' : 14,
#           'Link': 'www.weatherrescue.org', 
#           'Vbl': 'mslp', 'Stat': 'point', 'Units': 'hPa', 
#           'Meta': 'No idea yet', 
#           'Month': data[1,:], 'Year': data[0,:], 
#           'Day': data[2,:],
#           'Hour': data[3,:], 'Minute': data[4,:], 
#           'Period': perlist,
#           'Value': data[-1,:],
#           'Meta2': m2, 
#           'orig_time': ot
#           }
#
#obs = create(records)
#SEF.write_file(obs, sefdir+'test.tsv')