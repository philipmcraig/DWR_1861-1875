# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:43:31 2019

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
from hf_funcs import OSGB36toWGS84, WGS84toOSGB36
from functions import NearestIndex
#from mpl_toolkits.basemap import Basemap
import sklearn.metrics as skm
from netCDF4 import Dataset
import matplotlib.gridspec as gridspec

pl.close('all')
jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
raindir = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/60km/rainfall/day/v20181126/'

obs = pd.read_csv(jashome+'oct1903_rain_extra.csv',index_col=0)
stations = obs.columns.values
obs = pl.asarray(obs)

locs = pl.genfromtxt(jashome+'latlon_extras.txt',skip_header=0)
locs = locs[:,1:].copy()

ncfile = Dataset(ncasdir+'lons_lats_60km_osgb_grid.nc','r')
lons = ncfile.variables['longitudes'][:]
lats = ncfile.variables['latitudes'][:]
ncfile.close()

ncfile = Dataset(raindir+'rainfall_hadukgrid_uk_60km_day_19031001-19031031.nc','r')
rain = ncfile.variables['rainfall'][:]
ncfile.close()

rain = rain.data

x_coords = pl.arange(-200000+500,700001-500,1000)
y_coords = pl.arange(-200000+500,1250001-500,1000)

grid = pl.zeros_like(obs)

for stat in range(len(stations)):#
    loc = (locs[stat,1],locs[stat,0]) # lon,lat

    a = WGS84toOSGB36(loc[1],loc[0])
    ind1 = NearestIndex(x_coords,a[0]); ind2 = NearestIndex(y_coords,a[1])
    if rain[0,ind2,ind1] == rain.max():
        R = rain[0,ind2-5:ind2+6,ind1-5:ind1+6]
        N = pl.where(R!=R.max())
        d = pl.sqrt((4-N[0][:])**2+(4-N[1][:])**2)
        dm = pl.where(d==d.min())
        i = N[0][dm[0][0]]; j = N[1][dm[0][0]]
        latind = rain.shape[1] - ind2 - (i-5)
        lonind = ind1 - (5 - j)
    else:
        latind = rain.shape[1]-ind2; lonind = ind1
    
    grid[:,stat] = rain[:,-latind,lonind]

#pl.plot(obs*25.4)
#pl.plot(grid)
#fig, ax = pl.subplots(3,3,figsize=(10,10))
#
#for stat in range(len(stations)):
#    axx = pl.subplot(3,3,stat+1)
#    axx.plot(grid[:,stat],label='Met Office')
#    axx.plot(obs[:,stat]*25.4,label='Weather Rescue')
#    pl.xlim(0,30); axx.grid(axis='y',ls='--'); pl.ylim(0,40)
#    axx.set_xticklabels([1,6,11,16,21,26,31])
#    pl.title(stations[stat])
#    if stat in (0,3,6):
#        pl.ylabel('mm',fontsize=14)
#    if stat > 5:
#        pl.xlabel('days',fontsize=14)
#    if stat == 3:
#        axx.legend(loc=2)
#
#pl.tight_layout()

#dwr11yrs = pl.genfromtxt(ncasdir+'dwrrain_18stations_1900-1910.csv',dtype=None,
#                         delimiter=',')
#grid11yrs = pl.genfromtxt(ncasdir+'gridrain_18stations_1900-1910.csv',dtype=None,
#                          delimiter=',')
#
#stations = dwr11yrs[0]
#inds = [9,10,11,12,16]
#fig = pl.figure(figsize=(10,6))
#gs = gridspec.GridSpec(2,6)
#ig = [gs[0,0:2],gs[0,2:4],gs[0,4:],gs[1,1:3],gs[1,3:5]]
#
#for i in range(len(ig)):
#    ax = pl.subplot(ig[i])
#    ax.plot(grid11yrs[1368:1399,inds[i]],label='Met Office')
#    ax.plot(dwr11yrs[1369:1400,inds[i]].astype(float)*25.4,label='Weather Rescue')
#    pl.xlim(0,30); ax.set_xticklabels([1,6,11,16,21,26,31])
#    ax.grid(axis='y',ls='--'); pl.ylim(0,50)
#    pl.title(stations[inds[i]])
#    if i == 0:
#        pl.legend(loc=2)
#    if i in (0,3):
#        pl.ylabel('mm',fontsize=14)
#    if i > 2:
#        pl.xlabel('days',fontsize=14)
#
#pl.tight_layout()