# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:51:16 2019

@author: pmcraig
"""

import pylab as pl
import pandas as pd
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

pl.close('all')
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
ispddir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/ISPD/'
dwrdir = ncasdir + 'DWRcsv/'

year = 1903

pl.figure(figsize=(10,7.5))
ax = pl.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='50m')
ax.set_extent([-30,25,30,70],ccrs.PlateCarree())

land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
ax.add_feature(land_50m,alpha=0.5)


borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='lightgrey',
                                        facecolor=cfeature.COLORS['land'])
ax.add_feature(borders_50m,linewidth=0.5,zorder=5)#cfeature.BORDERS
ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['water'])
ax.add_feature(ocean_50m,alpha=0.5)

ispdfile = ispddir + 'ISPD47_'+str(year)+'.txt'
with open(ispdfile,'r') as f:
    ispddata = f.readlines()

ispd_lons = []; ispd_lats = []
for i in range(len(ispddata)):
    if ispddata[i][20:23] == '181' or ispddata[i][20:23] == '183':
        ispd_lons.append(float(ispddata[i][27:34]))
        ispd_lats.append(float(ispddata[i][35:40]))

ispd_locs = pl.array([pl.asarray(ispd_lons),pl.asarray(ispd_lats)]).T

ax.plot(ispd_locs[:,0],ispd_locs[:,1],marker='x',color='k',linewidth=0,zorder=15,
        transform=ccrs.PlateCarree(),alpha=0.25,ms=7.5,mew=2)

ax.annotate('ISPDv4.7 land stations in 1903',(-29,47),size=18,
            bbox={'boxstyle':'square','fc':'w'})

pl.tight_layout()
pl.savefig(ncasdir+'ISPD/map_ispd_land_1903_poster.png')