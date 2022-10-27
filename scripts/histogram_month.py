# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:38:25 2019

@author: pmcraig
"""

import pylab as pl
import xarray as xr
import glob
import pandas as pd
from scipy.stats import skew
from scipy.stats import normaltest, skewtest
import pcraig_funcs as pc

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'

year = '1902'
month = '03'

coords = (-3.090779,58.442519)
loc = 'Wick'

ncfile = xr.open_dataset(CR20dir+year+'/'+month+'/'+'prmsl_eu.nc')
data = xr.DataArray(ncfile.PRMSL_P1_L101_GGA0)
lon = xr.DataArray(ncfile.lon_0)
lat = xr.DataArray(ncfile.lat_0)

#monmean = pl.mean(data[:,3::4,0,:,:],axis=1)

statdata = pl.zeros([data.shape[0],data.shape[1],data.shape[2]])
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            statdata[i,j,k] = pc.BilinInterp(coords,lon,lat,data[i,j,k,:,:])

statdata_6pm = statdata[:,3::4,0]
statdata_9pm = statdata[:,3::4,1]
statdata_6am = statdata[:,1::4,0]
statdata_9am = statdata[:,1::4,1]
statdata_12pm = statdata[:,2::4,0]
statdata_3pm = statdata[:,2::4,1]

statdata_mor = statdata_6am + ((statdata_9am-statdata_6am)/(3*60*60))*(1*60*60)
statdata_eve = statdata_6pm + ((statdata_9pm-statdata_6pm)/(3*60*60))*(1*60*60)

statdata_all = pl.zeros([statdata_eve.shape[0],
                         statdata_eve.shape[1]+statdata_mor.shape[1]])
statdata_all[:,0::2] = statdata_mor[:]
statdata_all[:,1::2] = statdata_eve[:]

mn = pl.mean(statdata_all/100,axis=0)
sd = pl.std(statdata_all/100,axis=0)

dwrfiles = glob.glob(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'*')
dwr_feb = dwrfiles[59:90] # spans 1/2 to 1/3 for 6pm obs on 28/2

presobs_eve = pl.zeros([len(dwr_feb)-1])
presobs_mor = pl.zeros([len(dwr_feb)-1])
for i in range(1,len(dwr_feb)):
    logs = pd.read_csv(dwr_feb[i])
    obs = pl.array(logs)
    ind = pl.where(obs[:,0]==loc+' ')
    if obs[ind,1][0][0] == ' ' or obs[ind,1][0][0] == '  ':
        presobs_eve[i-1] = pl.float64('nan')
    else:
        presobs_eve[i-1] = pl.float64(obs[ind,1][0][0])
presobs_eve = presobs_eve*33.8639

for i in range(len(dwr_feb)-1):
    logs = pd.read_csv(dwr_feb[i])
    obs = pl.array(logs)
    ind = pl.where(obs[:,0]==loc+' ')
    if obs[ind,3][0][0] == ' ' or obs[ind,3][0][0] == '  ':
        presobs_mor[i] = pl.float64('nan')
    else:
        presobs_mor[i] = pl.float64(obs[ind,3][0][0])
presobs_mor = presobs_mor*33.8639

allpres = pl.zeros([len(presobs_mor)+len(presobs_eve)])
allpres[0::2] = presobs_mor[:]
allpres[1::2] = presobs_eve[:]

err = (mn-allpres)/sd
err2 = (mn-allpres)/pl.sqrt(sd**2 + 1.6**2)
err_mn = pl.nanmean(err)
err_sd = pl.nanstd(err); print 'standard deviation of z = ', err_sd
err2_sd = pl.nanstd(err2); print 'standard deviation of z2 = ', err2_sd


fig = pl.figure(); ax = pl.gca()
n, bins, patches = pl.hist(err[~pl.isnan(err)],ec='k',normed=1,zorder=2,bins=20)
y = pl.normpdf(bins,err_mn,err_sd)
pl.plot(bins,y,lw=2)
pl.axvline(0,ls='-',color='k',lw=1.5)
pl.axvline(err_mn,ls='--',color='k',lw=1.5)
pl.axvline(err_mn+err_sd,ls='--',color='k',lw=1.5)
pl.axvline(err_mn-err_sd,ls='--',color='k',lw=1.5)
#pl.xlim(-4,4)
pl.xlabel('z score')
pl.title(loc+' '+month+'/'+year)
pl.annotate('$\sigma_z$='+str(round(err_sd,2)),xy=(0.10,0.85),
            xycoords='figure fraction',fontsize=13)
pl.tight_layout()
#pl.savefig(jashome+'histograms/'+loc.replace(' ','')+'_hist_'+year+month+'.png')