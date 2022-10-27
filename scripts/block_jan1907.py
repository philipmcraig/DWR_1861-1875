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
    for i in range(pres.size):
        if pl.isnan(pres[i]) == True:
            pass
#        elif i == 10: # Liverpool
#           pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
#                        zorder=4,textcoords='offset pixels',xytext=(-17,3))
        elif i == 10: # Holyhead
            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
                        zorder=4,textcoords='offset pixels',xytext=(-20,3))
#        elif i == 14:
#            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
#                        zorder=4,textcoords='offset pixels',xytext=(-40,-6))
        elif i == 14: # Portland Bill
            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
                        zorder=4,textcoords='offset pixels',xytext=(3,-5))
        elif i == 15: # Dungeness
                pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
                        zorder=4,textcoords='offset pixels',xytext=(-15,-14))
#        elif i == 22:
#            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
#                        zorder=4,textcoords='offset pixels',xytext=(-15,4))
#        elif i == 25:
#            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
#                        zorder=4,textcoords='offset pixels',xytext=(-20,4))
        elif i == 26:
            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
                        zorder=4,textcoords='offset pixels',xytext=(-20,4))
        elif i == 27: # London
            pl.annotate(str(int(round(pres[i],0))),(coords[i,1],coords[i,0]),
                        zorder=4,textcoords='offset pixels',xytext=(-18,-14))
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

def ErrLabels(pres,ensmean,spread,coords,lon,lat):
    """
    """
    for i in range(len(pres)):
        if pl.isnan(pres[i]) == True:
            pass
        else:
            err = ErrorCalc(pres[i],ensmean[91,0]/100,spread[91,0]/100,
                                (coords[i,1],coords[i,0]),lon,lat)
            if i == 14: # Portland Bill
                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(3,-5))
            elif i == 10: # Holyhead
                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(-15,3))
            elif i == 15: # Dungeness
                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(-15,-14))
#            elif i == 16: # Dover
#                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
#                    zorder=4,textcoords='offset pixels',xytext=(-15,-10))
            elif i == 23:
                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(-15,4))
            elif i == 26: # Bath
                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(-15,4))
#            elif i == 27: # Oxford
#                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
#                    zorder=4,textcoords='offset pixels',xytext=(-15,4))
            elif i == 27: # London
                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(-22,-14))
            else:
                pl.annotate(str(round(err,2)),(coords[i,1],coords[i,0]),
                    zorder=4,textcoords='offset pixels',xytext=(2,-2))


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
#CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
CR20dir = ncasdir + '20CR/'
year = '1907'
month = '01'

data = pl.genfromtxt(ncasdir+'latlon_files/latlon'+year+'_pc.txt')
morstats = pl.where(((data[:,-4]==8) & (data[:,-3]==0))) # morning stations
evestats = pl.where((data[:,-2]==18) & (data[:,-1]==0)) # evening stations
#es1 = pl.where(data[:,-2]==18)
#es2 = pl.where(data[:,-2]==17)
#es1 = es1[0].tolist(); es1.remove(data.shape[0]-1)
#es1.extend(es2[0].tolist()); evestats = es1
evedata = data[evestats[0][:]]
#evedata = data[evestats]

obs = pl.genfromtxt(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+month+'_24.csv',delimiter=',')

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

#correct for gravity and change to hPa:
pres[0] = pres[0]+0.04 # Sumburgh Head
N = [1,2,3,18,19,20,21] # Malin Head/Shields and north (not Sumburgh Head)
pres[N] = pres[N] + 0.03
S = [4,5,6,7,8,9,10,11,14,15,16,17,22,23,24,25,26,27,28] # south of Malin/Shields
pres[S] = pres[S] + 0.02
pres[[12,13]] = pres[[12,13]] + 0.01 # Scilly & Jersey
pres = pres*33.8639


meanfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu_em_'+year+month+'.nc')
ensmean = xr.DataArray(meanfile.PRMSL_P1_L101_GGA0)
lat = xr.DataArray(meanfile.lat_0)
lon = xr.DataArray(meanfile.lon_0)
inittime = xr.DataArray(meanfile.initial_time0)
foretime = xr.DataArray(meanfile.forecast_time0)

sprdfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu_sprd_'+year+month+'.nc')
spread = xr.DataArray(sprdfile.PRMSL_sprd)

#pl.figure(figsize=(12.5,8))
fig, ax = pl.subplots(1,2,figsize=(19,6.7))

ax1 = pl.subplot(121,projection=ccrs.PlateCarree())#pl.axes(projection=ccrs.PlateCarree())
ax1.set_extent([-20.001,20.001,39.999,65],ccrs.PlateCarree())
ax1.coastlines(color='grey',resolution='50m',linewidth=0.5)
X, Y = pl.meshgrid(lon,lat)
cn = ax1.contour(X,Y,ensmean[91,0]/100,norm=pl.Normalize(972,1060),
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
cs = ax2.contourf(X,Y,spread[91,0]/100,norm=pl.Normalize(0,5),#colors='grey',
            levels=pl.linspace(1,5,9),alpha=0.4,cmap='OrRd',extend='both',
            transform=ccrs.PlateCarree())
#pl.colorbar(cs,extend='both',orientation='vertical')
#pl.clabel(cs,inline=True,fmt="%.1f",zorder=3,inline_spacing=5,manual=True)
ax2.plot(coords[:,1],coords[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())

ErrLabels(pres,ensmean,spread,coords,lon,lat)
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

time = inittime.values[91]
time = list(time)
time[0], time[3] = time[3], time[0]
time[1], time[4] = time[4], time[1]
time = ''.join(time)
pl.suptitle(time,y=0.99)

hour = time[-6:-4] + time[-3:-1]
day = time[:2]
savetime = year+month+day+'_'+hour
#pl.savefig(jashome+'/mslp_obs_1907_comp/mslp_'+savetime+'_obs_sprd.png')