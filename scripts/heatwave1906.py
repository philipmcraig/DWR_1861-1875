# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:50:43 2019

@author: pmcraig
"""
from __future__ import division
import pylab as pl
import xarray as xr
from netCDF4 import Dataset
import glob
#from mpl_toolkits.basemap import Basemap, shiftgrid
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import timeit
#import pandas as pd
import pcraig_funcs as pc

def MakeTime(inittime,foretime,ind,frc,H):
    """Create a string for time stamp for 20CR netcdf files
    
    Args:
        inittime (array): days in month x 4 6-hourly analysis steps; 
                          18 characters as individual array elements
                          MM/DD/YYYY (hh:00)
        foretime (array): 0 or 3, forecast step from analysis
        ind (int): index in inittime array
        frc (int): index in foretime array
        H (int): number of extra hours not on analysis or forecast steps
    """
    # add the forecast hours on to the analysis step hours
    #inititime[ind,-5] = str(int(inittime[ind,-5]) + foretime[frc])
    IT = list(inittime[ind])
    IT[0], IT[3] = IT[3], IT[0] # put the day and month in correct order
    IT[1], IT[4] = IT[4], IT[1]
    hour = str("{0:0=2d}".format(int(''.join(IT[-6:-4]))+H))#+foretime[frc]
    IT[-6] = hour[0]
    IT[-5] = hour[1]
#    inittime[0], inittime[3] = inittime[3], inittime[0]
#    inittime[1], inittime[4] = inittime[4], inittime[1]
    time = ''.join([str(x) for x in IT])
    
    return time

def MakeAverage(data_in,ind,frc,H):
    """
    """
    M = data_in[ind,0]
    M2 = data_in[ind,1]
    dp_dt = (M2-M)/(3*60*60)
    MI = M + dp_dt*(H*60*60)
    
    return MI

def PresLabels(presobs,coords):
    """
    Args:
        presobs (array): pressure observations
        coords (array): latitude, longitude co-ordinates of stations
    """
    for i in range(pres.size):
        if pl.isnan(pres[i]) == True:
            pass
        elif i == 14:
            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
                        zorder=4,textcoords='offset pixels',xytext=(-40,-6))
        elif i == 23:
            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
                        zorder=4,textcoords='offset pixels',xytext=(-15,4))
        else:
    		pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
                        zorder=4,textcoords='offset pixels',xytext=(2,-2))

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

def ErrLabels(varobs,varmean,varsprd,loc,lon,lat):
    """
    """
    errs = pl.zeros([len(varobs)])
    for i in range(len(varobs)):
        if pl.isnan(varobs[i]) == True:
            pass
        else:
            errs[i] = ErrorCalc(varobs[i],varmean,varsprd,loc[i],lon,lat)
            if i == 14:
                pl.annotate(str(int(round(varobs[i],0))),(loc[i,1],loc[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(-40,-6))
            elif i == 23:
                pl.annotate(str(int(round(varobs[i],0))),(loc[i,1],loc[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(-15,4))
            else:
                pl.annotate(str(int(round(varobs[i],0))),(loc[i,1],loc[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(2,-2))
    
    return None


pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
raindir = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/day/v20181126/'
#CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
CR20dir = ncasdir + '20CR/'
year = '1906'
month = '09'

#presfile = Dataset(CR20dir+year+'/'+month+'/prmsl_eu_em_'+year+month+'.nc','r')
#ensmean = presfile.variables['PRMSL_P1_L101_GGA0'][:]
#lat = presfile.variables['lat_0'][:]
#lon = presfile.variables['lon_0'][:]
#inittime = presfile.variables['initial_time0'][:]
#foretime = presfile.variables['forecast_time0'][:]
#presfile.close()
meanfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu_em_'+year+month+'.nc')
ensmean = xr.DataArray(meanfile.PRMSL_P1_L101_GGA0)
lat = xr.DataArray(meanfile.lat_0)
lon = xr.DataArray(meanfile.lon_0)
inittime = xr.DataArray(meanfile.initial_time0)
foretime = xr.DataArray(meanfile.forecast_time0)

#sprdfile = Dataset(CR20dir+year+'/'+month+'/prmsl_eu_sprd_'+year+month+'.nc','r')
#spread = sprdfile.variables['PRMSL_sprd'][:]
#sprdfile.close()
sprdfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu_sprd_'+year+month+'.nc')
spread = xr.DataArray(sprdfile.PRMSL_sprd)

skewfile = Dataset(CR20dir+year+'/'+month+'/mslp_eu_skew_'+year+month+'.nc','r')
skew = skewfile.variables['mslp skewness'][:]
skewfile.close()
#skewfile = xr.open_dataset(CR20dir+year+'/'+month+'/mslp_eu_skew_'+year+month+'.nc')
#skew = xr.DataArray(skewfile.mslp skewness)

data = pl.genfromtxt(ncasdir+'latlon_files/latlon1906_pc.txt')
morstats = pl.where(data[:,-4]==8) # morning stations
evestats = pl.where(data[:,-2]==18) # evening stations
#mordata = data[morstats[0][:]]
#evedata = data[evestats[0][:]]

obs = pl.genfromtxt(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+month+'_03.csv',delimiter=',')

WHICHOB = 1 # select morning (0), evening (1), Tmax (2) DWR observations

if WHICHOB == 0:
    pres = obs[morstats[0][:],3]
    coords = data[morstats[0][:]][:,1:3]
elif WHICHOB == 1:
    pres = obs[evestats[0][:],1]
    coords = data[evestats[0][:]][:,1:3]
elif WHICHOB == 2:
    pres = obs[:,-3]
    coords = data[:,1:3]
#evepres = obs[evestats[0][:-1],1]*33.8638
#morpres = obs[morstats[0][:],3]
#evepres = obs[evestats[0][:],1]#*33.8638
#correct for gravity and change to hPa:
pres[0] = pres[0]+0.04 # Sumburgh Head
N = [1,2,15,16,17,18,19] # Malin Head/Shields and north (not Sumburgh Head)
pres[N] = pres[N] + 0.03
S = [3,4,5,6,7,8,9,10,13,14,20,21,22,23,24,25,26] # south of Malin/Shields
pres[S] = pres[S] + 0.02
pres[[11,12]] = pres[[11,12]] + 0.01 # Scilly & Jersey
pres = pres*33.8639

sep3rd = pl.zeros([4,2,lat.size,lon.size])
sep3rd[:,0,:,:] = ensmean[13:17,1,:,:]
sep3rd[:,1,:,:] = ensmean[14:18,0,:,:]
sep3rd = pl.reshape(sep3rd,newshape=(8,lat.size,lon.size))
max3rd = pl.amax(sep3rd,axis=0)

ind = 7
frc = 0
H = 0
if frc == 0:
    MI = ensmean[ind,frc]
elif frc == 1:
    MI = MakeAverage(ensmean,ind,frc,H)
SI = MakeAverage(spread,ind,frc,H)
SK = MakeAverage(skew,ind,frc,H)
inittime = pl.asarray(inittime)
foretime = foretime.values.astype('timedelta64[h]').astype('int')
time  = MakeTime(inittime,foretime,ind,frc,H)
print time

pl.figure(figsize=(12.5,8))
ax = pl.axes(projection=ccrs.PlateCarree())
ax.set_extent([-15.0001,15.0001,39.999,65],ccrs.PlateCarree())
ax.coastlines(color='grey',resolution='50m',linewidth=0.5,zorder=1)
X, Y = pl.meshgrid(lon,lat)
#cm = pl.pcolormesh(X,Y,skew[ind,frc],norm=pl.Normalize(-1.5,1.5),alpha=0.5,
#                   transform=ccrs.PlateCarree(),linewidth=0,rasterized=True,
#                    antialiased=True,interpolation='nearest')
#cm.set_edgecolor('face')
#cm = ax.imshow(pl.flipud(skew[ind,frc]),norm=pl.Normalize(-1.5,1.5),alpha=0.5,
#               transform=ccrs.PlateCarree(),extent=[-30.0001,20.0001,39.999,65])
#pl.colorbar(cm)
#cs = pl.contourf(X,Y,max3rd-273.15,norm=pl.Normalize(10,30),alpha=0.5,
#	levels=pl.linspace(10,30,11),extend='both',cmap='YlOrRd',
#                    transform=ccrs.PlateCarree(),rasterized=True,antialiased=True)
cn = pl.contour(X,Y,SI/100,alpha=1.0,norm=pl.Normalize(0,6),#cmap='YlOrRd',
	antialiased=True,levels=[0,0.5,1,1.5,2,3,4],zorder=1,
                    transform=ccrs.PlateCarree(),linewidths=2.0)
#cb = pl.colorbar(cs)
#cb.set_label('$^\circ$C',fontsize=15)
pl.clabel(cn,inline=True,fmt="%.1f",zorder=2,inline_spacing=5,manual=True)
pl.title(time+' 20CR ensemble spread & z-scores at stations')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
gl.xlabels_top = False
gl.ylabels_left = True; gl.ylabels_right = True
gl.xlocator = mticker.FixedLocator([-40,-30,-20,-10,0,10,20,26,30])
gl.ylocator = mticker.FixedLocator([40,50,60,70])
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'color': 'k','size':11}
gl.ylabel_style = {'color': 'k','size':11}

ax.plot(coords[:,1],coords[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())
#PresLabels(pres,coords)
#ErrLabels(pres,ensmean[ind,frc],spread[ind,frc],pl.fliplr(coords),lon,lat)
##errs = pl.zeros([len(pres)])
for i in range(len(pres)):
    if pl.isnan(pres[i]) == True:
        pass
    else:
        err = ErrorCalc(pres[i],MI/100,SI/100,
                            (coords[i,1],coords[i,0]),lon,lat)
        if i == 14:
            pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                zorder=4,textcoords='offset pixels',xytext=(-40,-6))
        elif i == 23:
            pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                zorder=4,textcoords='offset pixels',xytext=(-15,4))
        else:
            pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                zorder=4,textcoords='offset pixels',xytext=(2,-2))
    

pl.tight_layout()
#pl.subplots_adjust(left=0.035,right=0.965)

year = time[6:10]; month = time[3:5]; day = time[0:2]
step = time[-6:-4]+time[-3:-1]
savetime = year+month+day+'_'+step
#pl.savefig(jashome+'mslp_spread_'+year+'/mslp_'+savetime+'_sprd.png')