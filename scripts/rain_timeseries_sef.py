# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:08:55 2019

@author: pmcraig
"""

import pylab as pl
import pandas as pd
import glob

def MetaData(origseries):
    """
    """
    import pylab as pl
    print "##### processing rain metadata #####"
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
            meta[i] = 'orig=' + str(round(origseries[i],2)) +'in'
        
#        if pl.isnan(origseries_mor[i])==True:
#            pass
#        else:
#            meta_mor[i] = 'orig=' + str(round(origseries_mor[i],2)) +'inHg'
    meta = list(meta)#; meta_mor = list(meta_mor)
    print "##### rain metadata completed #####"
    
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
sefdir = 'C:\Users\qx911590\Documents/Data_for_SEF/'#ncasdir + 'Data_for_SEF/'

years = pl.linspace(1861,1863,3).astype(int)
loc = 'Portrush'
vcol = -1 # variable column (1 for mslp, 2 for Td, -1 for rain)
datebreak = '1874/01/01'#'1864/07/01'#['1869/05/31','1872/03/22']##'1872/03/22'
varlist = []
datelist = []
yearlist = []
monthlist = []
daylist = []
metalist = []
origlist = []
count = 0

for Y in range(years.size):
    print years[Y]
    dwrfiles = glob.glob(dwrdir+'DWR_'+str(years[Y])+'*')
    
    #if yr 

    varyr = pl.zeros([len(dwrfiles)])
    origyr = pl.zeros([len(dwrfiles)])
    dates = pl.zeros([len(dwrfiles)],dtype='S10')
    for i in range(len(dwrfiles)):
        if years[Y]>=1872:
            df = pd.read_csv(dwrfiles[i],header=0,usecols=[0,1,2,3,4,5,6],
                           names=['station','pres','Td','Tw','Tx','Tn','rain',
                                              'pres_2pm','Td_2pm','Tw_2pm'])
        elif 1864<=years[Y]<=1871:
            df = pd.read_csv(dwrfiles[i],header=None,skiprows=1,usecols=[0,1,2,3])
        else:
            df = pd.read_csv(dwrfiles[i],header=None,skiprows=1)
        logs = pl.array(df)
        ind = pl.where(logs[:,0]==loc.upper())
        if ind[0].size == 1:
            ind = ind[0][0]
        elif ind[0].size == 0:
            varyr[i] = pl.float32('nan') # nan value for variable array
            origyr[i] = pl.float32('nan') # nan value for original array
            metalist.append('') # blank metadata entry
            continue
        
        if logs[ind,vcol] == ' ' or logs[ind,1] == '  ':
            varyr[i] = pl.float32('nan')
            origyr[i] = pl.float32('nan')
            metalist.append('')
        if logs[ind,vcol] in [999,'999',' 999.00',' 999', 'NaN']:
            varyr[i] = pl.float32('nan')
            origyr[i] = pl.float32('nan')
            metalist.append('')
        elif isinstance(logs[ind][vcol],str) == True and logs[ind][vcol][0] == '?':
            varyr[i] = float(logs[ind,vcol][1:])*25.4
            origyr[i] = float(logs[ind,vcol][1:])
            metalist.append('?')
        else:
            varyr[i] = float(logs[ind,vcol])*25.4
            origyr[i] = float(logs[ind,vcol])
            metalist.append('')
        
        yr = dwrfiles[i][55:59]
        mon = dwrfiles[i][60:62]
        day = dwrfiles[i][63:65]
        dates[i] = yr+'/'+mon+'/'+day
        yearlist.append(int(yr))
        monthlist.append(int(mon))
        daylist.append(int(day))
        
#        count = count + 1
#        if yr == '1908' and mon == '07' and day == '01':
#            A = count - 1
        
    varlist.append(varyr)
    origlist.append(origyr)
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
meta = pl.asarray(meta,dtype='S12')
hourseries = pl.array(hourseries)

# remove all values at indices where varseries has nan
varseries = pl.delete(varseries,X)
dateseries = pl.delete(dateseries,X)
metaseries = pl.delete(metaseries,X)
meta = pl.delete(meta,X)
hours = pl.delete(hourseries,X)

#for i in range(len(X)):
#    varseries.pop(X[i,0]-i)
#    dateseries.pop(X[i,0]-i)
#    yearlist.pop(X[i,0]-i)
#    monthlist.pop(X[i,0]-i)
#    daylist.pop(X[i,0]-i)
#    metalist.pop(X[i,0]-i)
#    meta.pop(X[i,0]-i)
#    
#varseries = pl.asarray(varseries)
#dateseries = pl.asarray(dateseries)
#yearseries = pl.asarray(yearlist)
#monthseries = pl.asarray(monthlist)
#dayseries = pl.asarray(daylist)
#metaseries = pl.asarray(metalist)
#meta = pl.asarray(meta)

QMs=pl.where(metaseries=='?')
for i in range(QMs[0].size):
    meta[QMs[0][i]] = meta[QMs[0][i]][:5] + metaseries[QMs[0][i]] + meta[QMs[0][i]][5:]

# use list comprehension to extract years, days & months:
yearseries = [int(i[:4]) for i in dateseries]
dayseries = [int(i[8:10]) for i in dateseries]
monthseries = [int(i[5:7]) for i in dateseries]

FRANCE = ['LORIENT','BREST','ROCHEFORT','CAPGRISNEZ','BIARRITZ','TOULON']

#hours = pl.zeros_like(dayseries)
mins = pl.zeros_like(dayseries)
if loc.upper() == 'HEARTSCONTENT':
    mins[:] = 30
    for i in range(len(dateseries)):
        if int(dateseries[i][:4]) == 1868 and int(dateseries[i][5:7]) == 12 and int(dateseries[i][8:10]) >= 2:
            cutoff = i
            break
    hours[cutoff:] = 12
elif loc.upper() == 'PARIS':
    start = pl.where(varseries!=0)[0][0] # where Paris rainfall starts
    yearseries = yearseries[start:]
    monthseries = monthseries[start:]
    dayseries = dayseries[start:]
    hours = hours[start:]
    mins = pl.full(shape=(len(yearseries),),fill_value=51)
    varseries = varseries[start:]
    meta = meta[start:]
elif loc.upper() in FRANCE:
    mins[:] = 51

if loc.upper() in ['PORTSMOUTH','PLYMOUTH','PENZANCE','HOLYHEAD','WICK']:
    split = pl.where(dateseries==datebreak)[0][0]
    data1 = pl.array([yearseries[:split],monthseries[:split],dayseries[:split],
                      hours[:split],mins[:split],varseries[:split],meta[:split]])
    data1 = pd.DataFrame(data1.T)
    
    # write data frame to .csv file
    data1.to_csv(sefdir+loc+'A_rain.csv',index=False,
                header=['year','month','day','hour','minute','rain','meta'])
    
    data2 = pl.array([yearseries[split:],monthseries[split:],dayseries[split:],
                      hours[split:],mins[split:],varseries[split:],meta[split:]])
    data2 = pd.DataFrame(data2.T)
    
    # write data frame to .csv file
    data2.to_csv(sefdir+loc+'B_rain.csv',index=False,
                header=['year','month','day','hour','minute','rain','meta'])
elif loc.upper() == 'LONDON':
    split1 = pl.where(dateseries==datebreak[0])[0][0]
    split2 = pl.where(dateseries==datebreak[1])[0][0]

    data1 = pl.array([yearseries[:split1],monthseries[:split1],dayseries[:split1],
                     hours[:split1],mins[:split1],varseries[:split1],meta[:split1]])
    data1 = pd.DataFrame(data1.T)
    data1.to_csv(sefdir+loc+'A_rain.csv',index=False,
                header=['year','month','day','hour','minute','rain','meta'])
    
    data2 = pl.array([yearseries[split1:split2],monthseries[split1:split2],
                      dayseries[split1:split2],hours[split1:split2],
                    mins[split1:split2],varseries[split1:split2],
                                                    meta[split1:split2]])
    data2 = pd.DataFrame(data2.T)
    data2.to_csv(sefdir+loc+'B_rain.csv',index=False,
                header=['year','month','day','hour','minute','rain','meta'])
    
    data3 = pl.array([yearseries[split2:],monthseries[split2:],dayseries[split2:],
                     hours[split2:],mins[split2:],varseries[split2:],meta[split2:]])
    data3 = pd.DataFrame(data3.T)
    data3.to_csv(sefdir+loc+'C_rain.csv',index=False,
                header=['year','month','day','hour','minute','rain','meta'])
elif loc.upper() == 'GREENCASTLE':
    split = pl.where(dateseries==datebreak)[0][0]
    
    data1 = pl.array([yearseries[:split],monthseries[:split],dayseries[:split],
                      hours[:split],mins[:split],varseries[:split],meta[:split]])
    data1 = pd.DataFrame(data1.T)
    # write data frame to .csv file
    data1.to_csv(sefdir+loc+'_rain.csv',index=False,
                header=['year','month','day','hour','minute','rain','meta'])
    
    
    data2 = pl.array([yearseries[split:],monthseries[split:],dayseries[split:],
                      hours[split:],mins[split:],varseries[split:],meta[split:]])
    data2 = pd.DataFrame(data2.T)
    
    # write data frame to .csv file
    data2.to_csv(sefdir+'Moville_rain.csv',index=False,
                header=['year','month','day','hour','minute','rain','meta'])
else:
    data = pl.array([yearseries,monthseries,dayseries,hours,mins,varseries,meta])
    data = pd.DataFrame(data.T)
    
    data.to_csv(sefdir+loc+'_rain.csv',index=False,#cols=header,
                header=['year','month','day','hour','minute','rain','meta'])