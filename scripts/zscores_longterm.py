# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:38:25 2019

@author: pmcraig
"""

import pylab as pl
import xarray as xr
import glob
import pandas as pd
import glob
from scipy.stats import skew
from scipy.stats import normaltest, skewtest
import timeit
import pcraig_funcs as pc

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
sefdir = '/badc/deposited2019/operation-weather-rescue/data/daily-weather-reports-1900-1910/SEF_files/'

year = '1903'
#month = '02'
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

coords = (-3.090779,58.442519) # lon, lat
loc = 'Wick'; ID = 'WICK'
GC = 0.03*33.8639 # gravity correction

statdata = []
inittime = []
start_time = timeit.default_timer()
for mt in range(len(months)):
    ncfile = xr.open_dataset(CR20dir+year+'/'+months[mt]+'/'+'prmsl_eu.nc')
    data = xr.DataArray(ncfile.PRMSL_P1_L101_GGA0)
    I = xr.DataArray(ncfile.initial_time0) 
    if mt == 0:
        lon = xr.DataArray(ncfile.lon_0)
        lat = xr.DataArray(ncfile.lat_0)
        foretime = xr.DataArray(ncfile.forecast_time0)
    ncfile.close()
    
    SD = pl.zeros([data.shape[0],data.shape[1],data.shape[2]])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                SD[i,j,k] = pc.BilinInterp(coords,lon,lat,data[i,j,k,:,:])

    SD = pl.reshape(SD,(SD.shape[0],SD.shape[1]*SD.shape[2]))
    statdata.append(SD)
    inittime.append(I.values)
elapsed = timeit.default_timer() - start_time
print elapsed

statdata = pl.concatenate(statdata,axis=1)#pl.ravel(pl.asarray(statdata))
iv = pl.concatenate(inittime)

#iv = inittime.values
fv = foretime.values
initmonths = [int(x[0:2]) for x in iv]
initdays = [int(x[3:5]) for x in iv]
inithours = [int(x[12:14]) for x in iv]
forehours = pl.array([fv[0],fv[1]],dtype='timedelta64[h]')

init3hours = pl.zeros([2*len(inithours)])
init3hours[0::2] = inithours[:]
foresteps = pl.asarray(inithours) + forehours[1].astype(int)
init3hours[1::2] = foresteps[:]

init3days = pl.zeros_like(init3hours)
init3days[0::2] = initdays[:]
init3days[1::2] = initdays[:]

init3months = pl.zeros_like(init3hours)
init3months[0::2] = initmonths[:]
init3months[1::2] = initmonths[:]

#statdata = pl.zeros([data.shape[0],data.shape[1],data.shape[2]])
#for i in range(data.shape[0]):
#    for j in range(data.shape[1]):
#        for k in range(data.shape[2]):
#            statdata[i,j,k] = pc.BilinInterp(coords,lon,lat,data[i,j,k,:,:])
#
#statdata = pl.reshape(statdata,(statdata.shape[0],statdata.shape[1]*statdata.shape[2]))

#statdata_6am = statdata[:,1::4,0]
#statdata_9am = statdata[:,1::4,1]
#statdata_12pm = statdata[:,2::4,0]
#statdata_3pm = statdata[:,2::4,1]
#statdata_6pm = statdata[:,3::4,0]
#statdata_9pm = statdata[:,3::4,1]
#
#statdata_mor = statdata_6am + ((statdata_9am-statdata_6am)/(3*60*60))*(1*60*60)
#statdata_eve = statdata_6pm + ((statdata_9pm-statdata_6pm)/(3*60*60))*(1*60*60)
#
#statdata_all = pl.zeros([statdata_eve.shape[0],
#                         statdata_eve.shape[1]+statdata_mor.shape[1]])
#statdata_all[:,0::2] = statdata_mor[:]
#statdata_all[:,1::2] = statdata_eve[:]
#
#mn = pl.mean(statdata_all/100,axis=0)
#sd = pl.std(statdata_all/100,axis=0)

#dwrfiles = glob.glob(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'*')
#dwr_feb = dwrfiles[59:90] # spans 1/2 to 1/3 for 6pm obs on 28/2
flist = glob.glob(sefdir+loc+'/'+'*'+'mslp'+'*')
F = pl.genfromtxt(flist[0],skip_header=13)
lines = pl.where(F[:,0]==int(year))#pl.where((F[:,0]==int(year)) & (F[:,1]==int(month)))
obsdata = F[lines[0],:-1]


obsdata[:,-1] = obsdata[:,-1] + GC


M = pl.zeros([obsdata.shape[0],statdata.shape[0]])

start_time = timeit.default_timer()
# loop over obsdata zero axis
for i in range(obsdata.shape[0]):
    # find where initdays = day & hour in obsdata
    X = pl.where((init3months==obsdata[i,1]) & (init3days==obsdata[i,2]) & (init3hours==obsdata[i,3]))
    # if X has zero size (hour not divisible by 3)
    if X[0].size == 0:
        H = int(obsdata[i,3]/3)
        D = pl.where(pl.asarray(init3days)==obsdata[i,2])

        mon = obsdata[i,1]
        if mon == 1:
            B = 0
        elif mon == 2:
            B = 31
        elif mon == 3:
            B = 59
        elif mon == 4:
            B = 90
        elif mon == 5:
            B = 120
        elif mon == 6:
            B = 151
        elif mon == 7:
            B = 181
        elif mon == 8:
            B = 212
        elif mon == 9:
            B = 243
        elif mon == 10:
            B = 273
        elif mon == 11:
            B = 304
        elif mon == 12:
            B = 334
            
        
        day = obsdata[i,2]
        #P = 8*(day-1)+H
        M1 = statdata[:,8*B+int(8*(day-1)+H)]/100
        M2 = statdata[:,8*B+int(8*(day-1)+H+1)]/100

        lower = init3hours[D[0][0]]
        diff = H - lower
        M[i] = M2 + ((M2-M1)/(3*60*60))*(diff*60*60)
        # take the two 3 hour steps on either side
        # interpolate to the time
            # statdata_minus3 + ((statdata_plus3 - statdata_minus3)/(3*60*60))*(H*60*60))
            # where H is the number of hours since t-3
    else:
        index = X[0][0]
        M[i] = statdata[:,index]/100
elapsed = timeit.default_timer() - start_time
print elapsed

Z = (pl.mean(M,axis=1)-obsdata[:,-1])/pl.std(M,axis=1)
Z2 = (pl.mean(M,axis=1)-obsdata[:,-1])/pl.sqrt(pl.std(M,axis=1)**2 + 1.6**2)

print 'Z = ', round(pl.std(Z),2)
print 'Z2 = ', round(pl.std(Z2),2)

pl.plot(obsdata[:,-1],label='obs')
pl.plot(pl.mean(M,axis=1),label='20CRv3')
pl.legend()

#for i in range(len(dwr_feb)-1):
#    logs = pd.read_csv(dwr_feb[i])
#    obs = pl.array(logs)
#    ind = pl.where(obs[:,0]==loc+' ')
#    if obs[ind,3][0][0] == ' ' or obs[ind,3][0][0] == '  ':
#        presobs_mor[i] = pl.float64('nan')
#    else:
#        presobs_mor[i] = pl.float64(obs[ind,3][0][0])
#presobs_mor = presobs_mor*33.8639
#
#allpres = pl.zeros([len(presobs_mor)+len(presobs_eve)])
#allpres[0::2] = presobs_mor[:]
#allpres[1::2] = presobs_eve[:]
#
#err = (mn-allpres)/sd
#err2 = (mn-allpres)/pl.sqrt(sd**2 + 1.6**2)
#err_mn = pl.nanmean(err)
#err_sd = pl.nanstd(err); print 'standard deviation of z = ', err_sd
#err2_sd = pl.nanstd(err2); print 'standard deviation of z2 = ', err2_sd
#
#
#fig = pl.figure(); ax = pl.gca()
#n, bins, patches = pl.hist(err[~pl.isnan(err)],ec='k',normed=1,zorder=2,bins=20)
#y = pl.normpdf(bins,err_mn,err_sd)
#pl.plot(bins,y,lw=2)
#pl.axvline(0,ls='-',color='k',lw=1.5)
#pl.axvline(err_mn,ls='--',color='k',lw=1.5)
#pl.axvline(err_mn+err_sd,ls='--',color='k',lw=1.5)
#pl.axvline(err_mn-err_sd,ls='--',color='k',lw=1.5)
##pl.xlim(-4,4)
#pl.xlabel('z score')
#pl.title(loc+' '+month+'/'+year)
#pl.annotate('$\sigma_z$='+str(round(err_sd,2)),xy=(0.10,0.85),
#            xycoords='figure fraction',fontsize=13)
#pl.tight_layout()
#pl.savefig(jashome+'histograms/'+loc.replace(' ','')+'_hist_'+year+month+'.png')
