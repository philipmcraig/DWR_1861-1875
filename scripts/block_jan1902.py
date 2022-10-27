# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:19:07 2018

@author: np838619
"""

import pylab as pl
from netCDF4 import Dataset
import glob
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import timeit
import matplotlib.ticker as mticker
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from adjustText import adjust_text
import pcraig_funcs as pc

def MakeTime(inittime,foretime,ind,frc):
    """Create a string for time stamp for 20CR netcdf files
    
    Args:
        inittime (array): days in month x 4 6-hourly analysis steps; 
                          18 characters as individual array elements
                          MM/DD/YYYY (hh:00)
        foretime (array): 0 or 3, forecast step from analysis
        ind (int): index in inittime array
        frc (int): index in foretime array
    """
    # add the forecast hours on to the analysis step hours
    #inititime[ind,-5] = str(int(inittime[ind,-5]) + foretime[frc])
    hour = str("{0:0=2d}".format(int(''.join(inittime[ind,-6:-4]))+foretime[frc]))
    inittime[ind,-6] = hour[0]
    inittime[ind,-5] = hour[1]
    time = ''.join([str(x) for x in inittime[ind]])
    
    return time

def PresLabels(presobs,coords):
    """
    Args:
        presobs (array): pressure observations
        coords (array): latitude, longitude co-ordinates of stations
    """
    p2 = pl.around(pres,0); p2 = p2[~pl.isnan(pres)].astype(int)
    crd_lon = coords[:,1][~pl.isnan(pres)]; crd_lat = coords[:,0][~pl.isnan(pres)]
    texts = [pl.text(crd_lon[i],crd_lat[i],p2[i].astype(str)) for i in range(len(p2))]
    adjust_text(texts)

    return None

def ErrorCalc(varobs,varmean,varsprd,loc,lon,lat):
    """
    varmean (array): ensemble mean
    varsprd (array): ensemble spread
    locs (array): longitude, latitude co-ordinates of stations
    lon (array): longitude array from 20CR
    lat (array): latitude array from 20CR
    """
    statmean = pc.BilinInterp(loc,lon,lat,varmean)
    statsprd = pc.BilinInterp(loc,lon,lat,varsprd)
    err = (statmean - varobs)/statsprd
    
    return err

def ErrLabels(pres,ensmean,spread,coords,lon,lat,ind):
    """
    """
    err = pl.zeros_like(pres)
    for i in range(len(pres)):
        if pl.isnan(pres[i]) == True:
            err[i] = pl.float32('nan')
        else:
            err[i] = ErrorCalc(pres[i],ensmean[ind,0]/100,spread[ind,0]/100,
                                (coords[i,1],coords[i,0]),lon,lat)

    e2 = pl.around(err,2); e2 = e2[~pl.isnan(err)]
    crd_lon = coords[:,1][~pl.isnan(err)]; crd_lat = coords[:,0][~pl.isnan(err)]
    texts = [pl.text(crd_lon[i],crd_lat[i],e2[i].astype(str)) for i in range(len(e2))]
    adjust_text(texts)

def GridLines(ax,top,left,right):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
    gl.xlabels_top = top
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([-40,-30,-20,-10,0,10,20,26,30])
    gl.ylocator = mticker.FixedLocator([40,50,60,70])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':11}
    gl.ylabel_style = {'color': 'k','size':11}
    
    return None
    
pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
#CR20dir = ncasdir + '20CR/'
year = '1902'
month = '01'

DATESEL = 31 # which day of the month?
HOURSEL = 18 # which hour of the day?
ind = (DATESEL-1)*4 + HOURSEL/6 # which index? doesn't work for 3 hour forecasts

data = pl.genfromtxt(ncasdir+'latlon_files/latlon'+year+'_pc.txt')
morstats = pl.where((data[:,-4]==8) & (data[:,-3]==0)) # morning stations

evestats = pl.where((data[:,-2]==18) & (data[:,-1]==0)) # evening stations
evefrance = pl.where((data[:,-2]==17) & (data[:,-1]==51)) # French evening stats
eveall = pl.concatenate((evestats[0],evefrance[0]))

#obs = pl.genfromtxt(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+'02'+'_01.csv',delimiter=',')

WHICHOB = 1 # select morning (0), evening (1), Tmax (2) DWR observations

if WHICHOB == 0:
    allstats = pl.where((HOURSEL-3<data[:,-4]) & (data[:,-4]<HOURSEL+3))
    obs = pl.genfromtxt(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+month+\
                        '_'+"{0:0=2d}".format(DATESEL)+'.csv',delimiter=',')
    pres = obs[allstats[0][:],3]
    coords = data[allstats[0][:]][:,1:3]
elif WHICHOB == 1:
    allstats = pl.where((HOURSEL-3<data[:,-2]) & (data[:,-2]<HOURSEL+3))
    obs = pl.genfromtxt(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+month+\
                        '_'+"{0:0=2d}".format(DATESEL+1)+'.csv',delimiter=',')
    pres = obs[allstats[0][:],1]
    coords = data[allstats[0][:]][:,1:3]
elif WHICHOB == 2:
    pres = obs[:,-3]
    coords = data[:,1:3]

#correct for gravity and change to hPa:
pres[9] = pres[9]+0.04 # Sumburgh Head
N = [10,11,24,25,26,27,28] # Malin Head/Shields and north (not Sumburgh Head)
pres[N] = pres[N] + 0.03
S = [12,13,14,15,16,17,18,19,22,23,29,30,31,32,33] # south of Malin/Shields
pres[S] = pres[S] + 0.02
pres[[20,21]] = pres[[20,21]] + 0.01 # Scilly & Jersey
pres = pres*33.8639


meanfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu.nc')
presdata = xr.DataArray(meanfile.PRMSL_P1_L101_GGA0)
lat = xr.DataArray(meanfile.lat_0)
lon = xr.DataArray(meanfile.lon_0)
inittime = xr.DataArray(meanfile.initial_time0)
foretime = xr.DataArray(meanfile.forecast_time0)

#sprdfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu.nc')
#spread = xr.DataArray(sprdfile.PRMSL_sprd)

ensmean = pl.mean(presdata,axis=0)
spread = pl.std(presdata,axis=0)

#pl.figure(figsize=(12.5,8))
fig, ax = pl.subplots(1,2,figsize=(19,6.7))

ax1 = pl.subplot(121,projection=ccrs.PlateCarree())#pl.axes(projection=ccrs.PlateCarree())
ax1.set_extent([-20.001,20.001,39.999,65],ccrs.PlateCarree())
ax1.coastlines(color='grey',resolution='50m',linewidth=0.5)
X, Y = pl.meshgrid(lon,lat)
cn = ax1.contour(X,Y,ensmean[ind,0]/100,norm=pl.Normalize(972,1060),
	colors='grey',levels=pl.linspace(972,1060,23),
                    alpha=0.5,transform=ccrs.PlateCarree())
#pl.colorbar(cs)
pl.clabel(cn,inline=True,fmt="%.0f",zorder=3,inline_spacing=5,manual=True)
ax1.plot(coords[:,1],coords[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())
PresLabels(pres,coords)


land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
ax1.add_feature(land_50m,alpha=0.5)
ax1.annotate('(a) 20CR ensemble mean & DWR observations',(-19.5,64),fontsize=12,
             bbox={'facecolor':'w'})

ax2 = pl.subplot(122,projection=ccrs.PlateCarree())
ax2.set_extent([-20.001,20.001,39.999,65],ccrs.PlateCarree())
ax2.coastlines(color='grey',resolution='50m',linewidth=0.5)
cs = ax2.contourf(X,Y,spread[ind,0]/100,norm=pl.Normalize(0,5),#colors='grey',
            levels=pl.linspace(1,5,9),alpha=0.4,cmap='OrRd',extend='both',
            transform=ccrs.PlateCarree())
#pl.colorbar(cs,extend='both',orientation='vertical')
#pl.clabel(cs,inline=True,fmt="%.1f",zorder=3,inline_spacing=5,manual=True)
ax2.plot(coords[:,1],coords[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())

ErrLabels(pres,ensmean,spread,coords,lon,lat,ind)
#ax2.add_feature(land_50m,alpha=0.5)
ax2.annotate('(b) 20CR ensemble spread & z-scores',(-19.5,64),fontsize=12,
             bbox={'facecolor':'w'})

GridLines(ax1,False,True,False)
GridLines(ax2,False,False,True)

pl.tight_layout()
pl.subplots_adjust(left=0.03,right=0.97,wspace=0.05,top=1.015,bottom=0.085)

f = pl.gcf()
colax = f.add_axes([0.6,0.07,0.3,0.03])                   
cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
cb.set_ticks(pl.linspace(1,5,9))
cb.set_label('hPa',fontsize=14)

time = inittime.values[ind]
time = list(time)
time[0], time[3] = time[3], time[0]
time[1], time[4] = time[4], time[1]
time = ''.join(time)
pl.suptitle(time,y=0.99)

hour = time[-6:-4] + time[-3:-1]
day = time[:2]
savetime = year+month+day+'_'+hour
#pl.savefig(jashome+'/mslp_obs_'+year+'_comp/mslp_'+savetime+'_obs_sprd.png')