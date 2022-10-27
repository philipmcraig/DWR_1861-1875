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
import pcraig_funcs as pc

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
sefdir = '/badc/deposited2019/operation-weather-rescue/data/daily-weather-reports-1900-1910/SEF_files/'

year = '1903'
month = '02'

coords = (-3.090779,58.442519)
loc = 'Wick'; ID = 'WICK'

ncfile = xr.open_dataset(CR20dir+year+'/'+month+'/'+'prmsl_eu.nc')
data = xr.DataArray(ncfile.PRMSL_P1_L101_GGA0)
lon = xr.DataArray(ncfile.lon_0)
lat = xr.DataArray(ncfile.lat_0)
inittime = xr.DataArray(ncfile.initial_time0)
foretime = xr.DataArray(ncfile.forecast_time0)
ncfile.close()
#monmean = pl.mean(data[:,3::4,0,:,:],axis=1)

iv = inittime.values
fv = foretime.values
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

statdata = pl.zeros([data.shape[0],data.shape[1],data.shape[2]])
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            statdata[i,j,k] = pc.BilinInterp(coords,lon,lat,data[i,j,k,:,:])

z = pl.reshape(statdata,(statdata.shape[0],statdata.shape[1]*statdata.shape[2]))

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
lines = pl.where((F[:,0]==int(year)) & (F[:,1]==int(month)))
obsdata = F[lines[0],:-1]

#X = pl.where((pl.asarray(initdays)==obsdata[1,2]) & (pl.asarray(inithours)==obsdata[1,3]))

M = pl.zeros([statdata.shape[0],obsdata.shape[0]])

# loop over obsdata zero axis
for i in range(obsdata.shape[0]):
    # find where initdays = day & hour in obsdata
    X = pl.where((pl.asarray(init3days)==obsdata[i,2]) & (pl.asarray(init3hours)==obsdata[i,3]))
    # if X has zero size (hour not divisible by 3)
    if X[0] == 0:
    	H = int(obsdata[i,3]/3) # if this is odd, use frc = 1 for lower bound
                             # if this is even, use frc = 0 for lower bound
    	day = obsdata[i,2]
	M1 = z[:,8*(day-1)+H]
	
        # take the two 3 hour steps on either side
        # interpolate to the time
            # statdata_minus3 + ((statdata_plus3 - statdata_minus3)/(3*60*60))*(H*60*60))
            # where H is the number of hours since t-3
    else:
        index = X[0][0]
        M[i] = z[:,index]
        # find index of hour with obsdata[i,3]/6
        #hour_index = int(obsdata[i,3]/6)
        # if divisible by 6 use the index & 0
        #if obsdata[i,3] % 6 == 0:
        #    fore_index = 0
        # if not divisible by 6 use the integer part of index & 1
        #else:
        #    fore_index = 1
        
        #day = obsdata[i,2]
        #M[i] = statdata[:,4*(day-1)+hour_index:4+4*(day-1)+hour_index,foreindex]
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
