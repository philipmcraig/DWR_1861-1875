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
    """Make the series of metadata for the station data, basically just the
    original values. This function has been modified from a previous version.
    
    Args:
        orig series: original values before conversion
    
    Returns:
        meta: original values with "orig=" and units to fit SEF style
    """
    #import pylab as pl
    print "##### processing mslp metadata #####"
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
            meta[i] = 'orig=' + str(origseries[i]) +'inHg'
        
#        if pl.isnan(origseries_mor[i])==True:
#            pass
#        else:
#            meta_mor[i] = 'orig=' + str(round(origseries_mor[i],2)) +'inHg'
    meta = list(meta)#; meta_mor = list(meta_mor)
    print "##### mslp metadata completed #####"
    
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

def CombineData(inseries_8am,inseries_2pm,split,dtype):
    """Combine the 8am and 2pm series where the dates match for the first time.
    
    Args:
        inseries_8am: regular 8am series
        inseries_2pm: additional 2pm series
        split: index where the dates match for the first time
        dtype: data type of outseries
    
    Returns:
        outseries: combined series as a list
    """
    holder = pl.zeros([len(inseries_8am)+len(inseries_2pm)],dtype=dtype)
    holder[:split+1] = inseries_8am[:split+1]
    holder[split+1::2] = inseries_2pm[:]
    holder[split+2::2] = inseries_8am[split+1:]
    
    outseries = list(holder)
    
    return outseries

#jashome = '/home/users/pmcraig/'
#ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
dwrdir = 'C:\Users\qx911590\Documents\dwr-1861-1875\\finished/'
sefdir = 'C:\Users\qx911590\Documents/Data_for_SEF/'#'M:\Data_for_SEF/'#ncasdir + 'Data_for_SEF/'

years = pl.linspace(1861,1863,3).astype(int) # years for which station data is available
loc = 'Portrush' # station name
vcol = 1 # variable column (1 for mslp, 2 for Td, -1 for rain)
vcol2pm = -3 # variable column for 2pm data (-3 for mslp, -2 for Td, -1 for Tw)
INC2PM = False # set to True to use 2pm data, False when there is none
datebreak = '1872/06/27'#['1869/05/31','1872/03/22']
# empty lists to append data to
varlist = []; var2pmlist = []
datelist = []; date2pmlist = []
#yearlist = []
#monthlist = []
#daylist = []
#hours = []
metalist = []
origlist = []; orig2pmlist = []
#count = 0

for Y in range(years.size): # loop over years
    print years[Y]
    dwrfiles = glob.glob(dwrdir+'DWR_'+str(years[Y])+'*') # all file for year

    # empty arrays for variable, original vals, dates to cover whole year
    varyr = pl.zeros([len(dwrfiles)])
    origyr = pl.zeros([len(dwrfiles)])
    dates = pl.zeros([len(dwrfiles)],dtype='S10')
    if INC2PM == True: # also set them up for 2pm data if required
        var2pmyr = pl.zeros([len(dwrfiles)])
        orig2pmyr = pl.zeros([len(dwrfiles)])
        dates2pm = pl.zeros([len(dwrfiles)],dtype='S10')
    
    for i in range(len(dwrfiles)): # loop over the files in a year
        if years[Y]>=1872: # 2pm data starts in 1872, so more columns in file
            df = pd.read_csv(dwrfiles[i],header=0, # read file as data frame
                           names=['station','pres','Td','Tw','Tx','Tn','rain',
                                              'pres_2pm','Td_2pm','Tw_2pm'])
        else: # all pre-1872 files are the same column for pressure
            df = pd.read_csv(dwrfiles[i],header=None,skiprows=1)
        logs = pl.array(df) # convert to numpy array
        ind = pl.where(logs[:,0]==loc.upper()) # find index of station
        if ind[0].size == 1:
            ind = ind[0][0]
        elif ind[0].size == 0: # if station is not in this file move to next
            varyr[i] = pl.float32('nan') # nan value for variable array
            origyr[i] = pl.float32('nan') # nan value for original array
            metalist.append('') # blank metadata entry
            continue
        
        if logs[ind,vcol] == ' ' or logs[ind,1] == '  ': # empty entries in file
            varyr[i] = pl.float32('nan') # set variable & original arrays to nan
            origyr[i] = pl.float32('nan')
            metalist.append('') # add blank to metalist
        elif logs[ind,vcol] in [999,'999',' 999.00',' 999',' NaN']:
            # different ways of showing no entry or invalid data
            varyr[i] = pl.float32('nan') # set variable & original arrays to nan
            origyr[i] = pl.float32('nan')
            metalist.append('') # add blank to metalist
        elif isinstance(logs[ind][vcol],str) == True and logs[ind][vcol][0] == '?':
            # file entry is a string and has ?
            varyr[i] = float(logs[ind,vcol][1:])*33.8639 # convert to hPa
            origyr[i] = float(logs[ind,vcol][1:]) # store original value
            metalist.append('?') # but only add ? to metalist
            print dwrfiles[i][51:]
        else: # otherwise these are the normal options
            varyr[i] = float(logs[ind,vcol])*33.8639
            origyr[i] = float(logs[ind,vcol])
            metalist.append('')
        
        # store date for each day in dates array
        yr = dwrfiles[i][55:59]
        mon = dwrfiles[i][60:62]
        day = dwrfiles[i][63:65]
        dates[i] = yr+'/'+mon+'/'+day
        
        # only use this if 2pm data exists
        if INC2PM == True and years[Y]>=1872:
            if logs[ind,vcol2pm] == ' ' or logs[ind,vcol2pm] == '  ': # empty entries in file
                var2pmyr[i] = pl.float16('nan') # set variable & original arrays to nan
                orig2pmyr[i] = pl.float16('nan')
                metalist.append('') # add blank to metalist
            elif logs[ind,vcol2pm] in [999,'999',' 999.00',' 999',' NaN']:
                # different ways of showing no entry or invalid data
                var2pmyr[i] = pl.float16('nan') # set variable & original arrays to nan
                orig2pmyr[i] = pl.float16('nan')
                metalist.append('') # add blank to metalist
            elif isinstance(logs[ind,vcol2pm],str) == True and logs[ind,vcol2pm][0] == '?':
                # file entry is a string and has ?
                var2pmyr[i] = float(logs[ind,vcol2pm][1:])*33.8639 # save as hPa
                orig2pmyr[i] = float(logs[ind,vcol2pm][1:]) # save original val
                metalist.append('?') # but only add ? to metalist
                print dwrfiles[i][51:]
            else: # otherwise these are the normal options
                var2pmyr[i] = float(logs[ind,vcol2pm])*33.8639
                orig2pmyr[i] = float(logs[ind,vcol2pm])
                metalist.append('')
            
            if i == 0: # 1st file of year needs last file of previous year for 2pm date
                dwrprev = glob.glob(dwrdir+'DWR_'+str(years[Y]-1)+'*')
                yr2pm = dwrprev[-1][55:59]
                mon2pm = dwrprev[-1][60:62]
                day2pm = dwrprev[-1][63:65]
            else: # otherwise use the previous day's filename
                yr2pm = dwrfiles[i-1][55:59]
                mon2pm = dwrfiles[i-1][60:62]
                day2pm = dwrfiles[i-1][63:65]
            # store date for each day in dates array
            dates2pm[i] = yr2pm+'/'+mon2pm+'/'+day2pm
    
    # add the variable, original value & date arrays to lists
    varlist.append(varyr)
    origlist.append(origyr)
    datelist.append(dates)
    
    if INC2PM == True and years[Y]>=1872: # do the same for 2pm data if present
        var2pmlist.append(var2pmyr)
        orig2pmlist.append(orig2pmyr)
        date2pmlist.append(dates2pm)

# combine into individual long arrays then convert to lists
varseries = pl.concatenate(varlist).ravel(); varseries = list(varseries)
origseries = pl.concatenate(origlist).ravel(); origseries = list(origseries)
dateseries = pl.concatenate(datelist).ravel(); dateseries = list(dateseries)

# array for hours of 8am data, same size as varseries
hourseries = pl.full(shape=(len(varseries),),fill_value=8).tolist()

if INC2PM == True: # same needed for 2pm data if it exists
    var2pmseries = pl.concatenate(var2pmlist).ravel(); var2pmseries = list(var2pmseries)
    orig2pmseries = pl.concatenate(orig2pmlist).ravel(); orig2pmseries = list(orig2pmseries)
    date2pmseries = pl.concatenate(date2pmlist).ravel(); date2pmseries = list(date2pmseries)
    
    hour2pmseries = pl.full(shape=(len(var2pmseries),),fill_value=14).tolist()
    
    # find the index where dates match in dateseries & date2pmseries
    split = pl.where(pl.array(dateseries)==date2pmseries[0])[0][0]
    
    # merge the standard and 2pm series at the matching index
    varseries = CombineData(varseries,var2pmseries,split,float)
    origseries = CombineData(origseries,orig2pmseries,split,float)
    dateseries = CombineData(dateseries,date2pmseries,split,'S10')
    hourseries = CombineData(hourseries,hour2pmseries,split,float)

meta = MetaData(origseries) # make into SEF style

# find where the nan values are in varseries
X = pl.argwhere(pl.isnan(varseries))
X = X.flatten() # flatten array to make it easier to work with

# convert lists to arrays to remove nan value indices
varseries = pl.asarray(varseries)
dateseries = pl.asarray(dateseries)
metaseries = pl.asarray(metalist)
meta = pl.asarray(meta,dtype='S15')
hourseries = pl.array(hourseries)

# remove all values at indices where varseries has nan
varseries = pl.delete(varseries,X)
dateseries = pl.delete(dateseries,X)
metaseries = pl.delete(metaseries,X)
meta = pl.delete(meta,X)
hours = pl.delete(hourseries,X)

# where is varseries = 0?
#Z = pl.where(varseries==0)[0]
#
##remove all values where varseries=0:
#varseries = pl.delete(varseries,Z)
#dateseries = pl.delete(dateseries,Z)
#metaseries = pl.delete(metaseries,Z)
#meta = pl.delete(meta,Z)
#hours = pl.delete(hours,Z)

# find the ? entries in the metaseries array
QMs=pl.where(metaseries=='?')
# combine entries in meta & metaseries to have e.g. "orig=?30.55F"
for i in range(QMs[0].size): # loop over QMs array
    meta[QMs[0][i]] = meta[QMs[0][i]][:5] + metaseries[QMs[0][i]] + meta[QMs[0][i]][5:]

# use list comprehension to extract years, days & months:
yearseries = [int(i[:4]) for i in dateseries]
dayseries = [int(i[8:10]) for i in dateseries]
monthseries = [int(i[5:7]) for i in dateseries]

FRANCE = ['PARIS','LORIENT','BREST','BAYONNE','ROCHEFORT','CAPGRISNEZ',
              'STRASBOURG','LYONS','BIARRITZ','TOULON','CHARLEVILLE']

mins = pl.zeros_like(dayseries) # zeros array for minutes, same size as others
if loc.upper() == 'HEARTSCONTENT':
    mins[:] = 30
    for i in range(len(dateseries)):
        if int(dateseries[i][:4]) == 1868 and int(dateseries[i][5:7]) == 12 and int(dateseries[i][8:10]) >= 2:
            cutoff = i
            break
    hours[cutoff:] = 12
elif loc.upper() in FRANCE:
    mins[:] = 51
elif loc.upper() in ['LISBON','OPORTO']:
    mins[:] = 36

if loc.upper() in ['PLYMOUTH','PORTSMOUTH','PENZANCE','HOLYHEAD','WICK']:
    split = pl.where(dateseries==datebreak)[0][0]
    data1 = pl.array([yearseries[:split],monthseries[:split],dayseries[:split],
                      hours[:split],mins[:split],varseries[:split],meta[:split]])
    data1 = pd.DataFrame(data1.T)
    
    # write data frame to .csv file
    data1.to_csv(sefdir+loc+'A_mslp.csv',index=False,
                header=['year','month','day','hour','minute','mslp','meta'])
    
    data2 = pl.array([yearseries[split:],monthseries[split:],dayseries[split:],
                      hours[split:],mins[split:],varseries[split:],meta[split:]])
    data2 = pd.DataFrame(data2.T)
    
    # write data frame to .csv file
    data2.to_csv(sefdir+loc+'B_mslp.csv',index=False,
                header=['year','month','day','hour','minute','mslp','meta'])
elif loc.upper() == 'LONDON':
    split1 = pl.where(dateseries==datebreak[0])[0][0]
    split2 = pl.where(dateseries==datebreak[1])[0][0]

    data1 = pl.array([yearseries[:split1],monthseries[:split1],dayseries[:split1],
                     hours[:split1],mins[:split1],varseries[:split1],meta[:split1]])
    data1 = pd.DataFrame(data1.T)
    data1.to_csv(sefdir+loc+'A_mslp.csv',index=False,
                header=['year','month','day','hour','minute','mslp','meta'])
    
    data2 = pl.array([yearseries[split1:split2],monthseries[split1:split2],
                      dayseries[split1:split2],hours[split1:split2],
                    mins[split1:split2],varseries[split1:split2],
                                                    meta[split1:split2]])
    data2 = pd.DataFrame(data2.T)
    data2.to_csv(sefdir+loc+'B_mslp.csv',index=False,
                header=['year','month','day','hour','minute','mslp','meta'])
    
    data3 = pl.array([yearseries[split2:],monthseries[split2:],dayseries[split2:],
                     hours[split2:],mins[split2:],varseries[split2:],meta[split2:]])
    data3 = pd.DataFrame(data3.T)
    data3.to_csv(sefdir+loc+'C_mslp.csv',index=False,
                header=['year','month','day','hour','minute','mslp','meta'])
elif loc.upper() == 'GREENCASTLE':
    split = pl.where(dateseries==datebreak)[0][0]
    
    data1 = pl.array([yearseries[:split],monthseries[:split],dayseries[:split],
                      hours[:split],mins[:split],varseries[:split],meta[:split]])
    data1 = pd.DataFrame(data1.T)
    # write data frame to .csv file
    data1.to_csv(sefdir+loc+'_mslp.csv',index=False,
                header=['year','month','day','hour','minute','mslp','meta'])
    
    
    data2 = pl.array([yearseries[split:],monthseries[split:],dayseries[split:],
                      hours[split:],mins[split:],varseries[split:],meta[split:]])
    data2 = pd.DataFrame(data2.T)
    
    # write data frame to .csv file
    data2.to_csv(sefdir+'Moville_mslp.csv',index=False,
                header=['year','month','day','hour','minute','mslp','meta'])
else:
    # stick everything into one array & convert to data frame
    data = pl.array([yearseries,monthseries,dayseries,hours,mins,varseries,meta])
    data = pd.DataFrame(data.T)
    
    # write data frame to .csv file
    data.to_csv(sefdir+loc+'_mslp.csv',index=False,
                header=['year','month','day','hour','minute','mslp','meta'])