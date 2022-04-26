# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:08:55 2019

@author: pmcraig
"""

import pylab as pl
import pandas as pd
import glob

def Get2pmData(dwrdir,locstring,varstring):
    """
    """
    import pylab as pl
    import pandas as pd
    
    df = pd.read_csv(dwrdir+'1900_2pm.csv',header=0)
    S = pl.where(df.columns.values==locstring+' '+varstring)
    if S[0].size == 1:
        print "##### 2pm data for " + locstring + " mslp exists #####"
        data2pm = pl.array(df)
        pres2pm = (pl.array(df[locstring+' '+varstring])[:-1])*33.8639
        orig2pm = pl.array(df[locstring+' '+varstring])[:-1]
        years2pm = pl.array([x[-4:] for x in data2pm[:-1,0]])
        months2pm = pl.array([x[3:5] for x in data2pm[:-1,0]])
        days2pm = pl.array([x[:2] for x in data2pm[:-1,0]])
        hours2pm = pl.zeros_like(pres2pm); hours2pm[:] = 14
        meta2pm = pl.zeros_like(pres2pm,dtype='object')
        for j in range(len(meta2pm)):
            if pl.isnan(orig2pm[j])==True:
                meta2pm[j] = 'nan'
            else:
                meta2pm[j] = 'orig='+str(round(orig2pm[j],2))+'inHg'
        
        return S, pres2pm, meta2pm, years2pm, months2pm, days2pm, hours2pm
    else:
        print "##### no 2pm data for " + locstring + " mslp #####"
        return S, [], [], [], [], [], []

def MetaData(origseries):
    """
    """
    import pylab as pl
    print "##### processing Tmax metadata #####"
    meta = pl.zeros([len(origseries)],dtype='object')
    #meta_mor = pl.zeros([len(origseries_mor)],dtype='object')
#    if splitpoint == None:
#        meta_eve[:] = 'PGC=N'; meta_mor[:] = 'PGC=N'
#    else:
#    meta_eve[:splitpoint] = 'PGC=N|'; meta_eve[splitpoint:] = 'PGC=Y|'
#    meta_mor[:splitpoint] = 'PGC=N|'; meta_mor[splitpoint:] = 'PGC=Y|'
    for i in range(meta.size):
        if pl.isnan(origseries[i])== True:
            pass
        else:
            meta[i] = 'orig=' + str(origseries[i]) +'F'
        
#        if pl.isnan(origseries_mor[i])==True:
#            pass
#        else:
#            meta_mor[i] = 'orig=' + str(round(origseries_mor[i],2)) +'inHg'
    meta = list(meta)#; meta_mor = list(meta_mor)
    print "##### Tmax metadata completed #####"
    
    return meta

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

#jashome = '/home/users/pmcraig/'
#ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
dwrdir = 'C:\Users\qx911590\Documents\dwr-1861-1875\\finished/'
sefdir = 'C:\Users\qx911590\Documents/Data_for_SEF/' #'M:\Data_for_SEF/'#ncasdir + 'Data_for_SEF/'

years = pl.linspace(1872,1875,4).astype(int)
loc = 'Shields'
vcol = 4 # variable column (1 for mslp, 2 for Td, 4 for Tmax, -1 for rain)
datebreak = '1872/06/27'
varlist = []
datelist = []
#yearlist = []
#monthlist = []
#daylist = []
metalist = []
origlist = []
#count = 0

for Y in range(years.size):
    print years[Y]
    dwrfiles = glob.glob(dwrdir+'DWR_'+str(years[Y])+'*')

    varyr = pl.zeros([len(dwrfiles)])
    origyr = pl.zeros([len(dwrfiles)])
    dates = pl.zeros([len(dwrfiles)],dtype='S10')
    for i in range(len(dwrfiles)):
        if years[Y]>=1872:
            df = pd.read_csv(dwrfiles[i],header=0,
                           names=['station','pres','Td','Tw','Tx','Tn','rain',
                                              'pres_2pm','Td_2pm','Tw_2pm'])
        #else:
        #    df = pd.read_csv(dwrfiles[i],header=0)
        logs = pl.array(df)
        ind = pl.where(logs[:,0]==loc.upper())
        if ind[0].size == 1:
            ind = ind[0][0]
        elif ind[0].size == 0:
            varyr[i] = pl.float32('nan') # nan value for variable array
            origyr[i] = pl.float32('nan') # nan value for original array
            metalist.append('') # blank metadata entry
            continue
        
        if logs[ind,vcol] == ' ' or logs[ind,vcol] == '  ':
            varyr[i] = pl.float32('nan')
            origyr[i] = pl.float32('nan')
            metalist.append('')
        if logs[ind,vcol] in [999,'999',' 999.00',' 999',' NaN']:
            varyr[i] = pl.float32('nan')
            origyr[i] = pl.float32('nan')
            metalist.append('')
        elif isinstance(logs[ind][vcol],str) == True and logs[ind][vcol][0] == '?':
            varyr[i] = (int(logs[ind,vcol][1:])-32)/1.8
            origyr[i] = int(logs[ind,vcol][1:])
            metalist.append('?')
        else:
            varyr[i] = (int(logs[ind,vcol])-32)/1.8
            origyr[i] = int(logs[ind,vcol])
            metalist.append('')
        
        yr = dwrfiles[i][55:59]
        mon = dwrfiles[i][60:62]
        day = dwrfiles[i][63:65]
        dates[i] = yr+'/'+mon+'/'+day
#        yearlist.append(int(yr))
#        monthlist.append(int(mon))
#        daylist.append(int(day))
        
#        count = count + 1
#        if yr == '1908' and mon == '07' and day == '01':
#            A = count - 1
        
    varlist.append(varyr)
    origlist.append(origyr.astype(int))
    datelist.append(dates)

varseries = pl.concatenate(varlist).ravel(); varseries = list(varseries)
origseries = pl.concatenate(origlist).ravel(); origseries = list(origseries)
dateseries = pl.concatenate(datelist).ravel(); dateseries = list(dateseries)

# array for hours of 8am data, same size as varseries
hourseries = pl.full(shape=(len(varseries),),fill_value=8).tolist()

meta = MetaData(origseries)

X = pl.argwhere(pl.isnan(varseries))
X = X.flatten() # flatten array to make it easier to work with

# convert lists to arrays to remove nan value indices
varseries = pl.asarray(varseries)
dateseries = pl.asarray(dateseries)
metaseries = pl.asarray(metalist)
meta = pl.asarray(meta,dtype='S9')
hourseries = pl.array(hourseries)

# remove all values at indices where varseries has nan
varseries = pl.delete(varseries,X)
dateseries = pl.delete(dateseries,X)
metaseries = pl.delete(metaseries,X)
meta = pl.delete(meta,X)
hours = pl.delete(hourseries,X)

QMs=pl.where(metaseries=='?')
for i in range(QMs[0].size):
    meta[QMs[0][i]] = meta[QMs[0][i]][:5] + metaseries[QMs[0][i]] + meta[QMs[0][i]][5:]

# use list comprehension to extract years, days & months:
yearseries = [int(i[:4]) for i in dateseries]
dayseries = [int(i[8:10]) for i in dateseries]
monthseries = [int(i[5:7]) for i in dateseries]

FRANCE = ['PARIS','LORIENT','BREST','ROCHEFORT','CAPGRISNEZ','BIARRITZ','TOULON']

mins = pl.zeros_like(dayseries)
if loc.upper() in FRANCE:
    mins[:] = 51


if loc.upper() in ['LONDON','PLYMOUTH','HOLYHEAD','WICK']:
    split = pl.where(dateseries==datebreak)[0][0]
    data1 = pl.array([yearseries[:split],monthseries[:split],dayseries[:split],
                      hours[:split],mins[:split],varseries[:split],meta[:split]])
    data1 = pd.DataFrame(data1.T)
    
    # write data frame to .csv file
    data1.to_csv(sefdir+loc+'A_tmax.csv',index=False,
                header=['year','month','day','hour','minute','tmax','meta'])
    
    data2 = pl.array([yearseries[split:],monthseries[split:],dayseries[split:],
                      hours[split:],mins[split:],varseries[split:],meta[split:]])
    data2 = pd.DataFrame(data2.T)
    
    # write data frame to .csv file
    data2.to_csv(sefdir+loc+'B_tmax.csv',index=False,
                header=['year','month','day','hour','minute','tmax','meta'])
if loc.upper() == 'GREENCASTLE':
    data = pl.array([yearseries,monthseries,dayseries,hours,mins,varseries,meta])
    data = pd.DataFrame(data.T)
    
    data.to_csv(sefdir+'Moville_tmax.csv',index=False,#cols=header,
                header=['year','month','day','hour','minute','tmax','meta'])
else:
    data = pl.array([yearseries,monthseries,dayseries,hours,mins,varseries,meta])
    data = pd.DataFrame(data.T)
    
    data.to_csv(sefdir+loc+'_tmax.csv',index=False,#cols=header,
                header=['year','month','day','hour','minute','tmax','meta'])
    #pl.savetxt(sefdir+loc+'_mslp.csv',data.T,delimiter=',')