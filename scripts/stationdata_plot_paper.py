# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:20:00 2020

@author: pmcraig
"""

from __future__ import division
import pylab as pl
import pandas as pd
import xarray as xr
from scipy import stats
import glob
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath
import pcraig_funcs as pc

def GridLines(ax,top,bottom,left,right):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
    gl.xlabels_top = top; gl.xlabels_bottom = bottom
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([18,20,25,30,32])
    gl.ylocator = mticker.FixedLocator([20,45,48,50])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':10}
    gl.ylabel_style = {'color': 'k','size':10}
    
    return None

def dmslat_conv(s):
    degrees = s[1:3]
    minutes = s[4:6]
    seconds = s[7:]

    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    
    return dd

def dmslon_conv(s):
    degrees = s[2:4]
    minutes = s[5:7]
    seconds = s[8:]
    
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if s[0] == '-':
        dd = -1*dd
    
    return dd

def Nonblended(dirpath,cn_name):
    """Read the ECA&D non-blended station data.
    
    Args:
        dirpath (string): directory containing data
        cn_name (string): two-character ISO country code in caps
    
    Returns:
        ta_o (array): mean growing season temperature, stations x years
    """
    # metadata from sources.txt
    df = pd.read_csv(dirpath+'TG_nonblended/sources.txt',header=20)
    data = pl.asarray(df)
    
    country = pl.where(data[:,2]==cn_name)[0] # all stations for this country
    
    stations = data[country,1] # select the station names
    # remove trailing blank spaces from the end of string:
    stations = pl.asarray([i.strip() for i in stations])
    print stations
    
    sid = data[country,0] # station ID numbers
    
    # extract station co-ordinates & convert to decimal form
    lats = pl.asarray([dmslat_conv(i) for i in data[country,3]])
    lons = pl.asarray([dmslon_conv(i) for i in data[country,4]])
    
    ta_o = pl.zeros([len(stations),68]) # 68 years in INDECIS
    
    for s in range(len(stations)):
        SOUID = sid[s] # SOUID is the ID used in nonblended station files
        statdata = pd.read_csv(dirpath+'TG_nonblended/TG_SOUID'+str(SOUID)+'.txt',skiprows=18)
        statdata = pd.DataFrame(statdata)
        
        temp = statdata[statdata.columns[3]]
        missing = pl.where(temp==-9999)[0]
        temp[missing] = pl.float32('nan')
        temp = temp/10.
        
        date = statdata[statdata.columns[2]]
        d = pl.array(date)
        years = [str(i)[:4] for i in d]; years = pl.array(years).astype(int)
        months = [str(i)[4:6] for i in d]; months = pl.array(months).astype(int)
        days = [str(i)[6:] for i in d]; days = pl.array(days).astype(int)
        
        if years.min()<=1950 and years.max()>=2017:
            y1 = pl.where(years==1950)[0][0]
            y2 = pl.where(years==2017)[0][-1]
            Y = pl.linspace(1950,2017,68)
        elif years.min()==years.max() and years.max()>2017:
            pass
        elif years.min()>1950 and years.max()>=2017:
            y1 = pl.where(years==years.min())[0][0]
            y2 = pl.where(years==2017)[0][-1]
            Y = pl.linspace(years.min(),2017,2017-years.min()+1)
        elif years.min()<=1950 and years.max()<2017:
            y1 = pl.where(years==1950)[0][0]
            y2 = pl.where(years==years.max())[0][-1]
            Y = pl.linspace(1950,years.max(),years.max()-1950+1)
        elif years.min()>1950 and years.max()<2017:
            y1 = pl.where(years==years.min())[0][0]
            y2 = pl.where(years==years.max())[0][-1]
            Y = pl.linspace(years.min(),years.max(),years.max()-years.min()+1)
        
        fullyears = pl.linspace(1950,2017,68)
        Z = len(fullyears)-len(Y)
        monthly_means = pl.zeros([68,12]); monthly_means[:,:] = pl.float32('nan')
        
        if years.min()==years.max() and years.max()>2017:
            ta_o[s,:] = pl.float32('nan')
        else:
            for yr in range(len(Y)):
                #Z = pl.where(fullyears==Y[yr])[0][0]
                for mn in range(1,13):
                    #Y = years[y1:y2][yr]
                    #M = months[y1:y2][mn]
                    X = pl.where((years[y1:y2+1]==Y[yr]) & (months[y1:y2+1]==mn))[0]
                    monthly_means[yr+Z,mn-1] = pl.mean(temp[y1:y2+1][X[0]:X[-1]+1])
            
            ta_o[s,:] = pl.nanmean(monthly_means[:,3:10],axis=1)
        #W = pl.where(pl.isnan(ta_o[s,:11]==False))
        #if W[0].size > 0:
        #    print stations[s]

    return ta_o, lats, lons, stations

def Blended(dirpath,cn_name):
    """Read the ECA&D blended station data
    """
    # metadata from stations.txt
    df = pd.read_csv(dirpath+'TG_nonblended/stations.txt',header=20)
    data = pl.asarray(df)
    
    country = pl.where(data[:,2]==cn_name)[0] # all stations for this country
    
    stations = data[country,1] # select the station names
    # remove trailing blank spaces from the end of string:
    stations = pl.asarray([i.strip() for i in stations])
    
    sid = data[country,0] # station ID numbers
    
    # extract station co-ordinates & convert to decimal form
    lats = pl.asarray([dmslat_conv(i) for i in data[country,3]])
    lons = pl.asarray([dmslon_conv(i) for i in data[country,4]])
    
    ta_o = pl.zeros([len(stations),68])
    
    for s in range(len(stations)):
        STAID = str(sid[s]).zfill(6) # add in the leading zeros
        statdata = pd.read_csv(indecis+'TG_blended/TG_STAID'+str(STAID)+'.txt',skiprows=20)
        statdata = pd.DataFrame(statdata)
        
        temp = statdata[statdata.columns[3]]
        missing = pl.where(temp==-9999)[0]
        temp[missing] = pl.float32('nan')
        temp = temp/10.
        
        date = statdata[statdata.columns[2]]
        d = pl.array(date)
        years = [str(i)[:4] for i in d]; years = pl.array(years).astype(int)
        months = [str(i)[4:6] for i in d]; months = pl.array(months).astype(int)
        days = [str(i)[6:] for i in d]; days = pl.array(days).astype(int)
        
        if years.min()<=1950 and years.max()>=2017:
            y1 = pl.where(years==1950)[0][0]
            y2 = pl.where(years==2017)[0][-1]
            Y = pl.linspace(1950,2017,68)
        elif years.min()==years.max() and years.max()>2017:
            pass
        elif years.min()>1950 and years.max()>=2017:
            y1 = pl.where(years==years.min())[0][0]
            y2 = pl.where(years==2017)[0][-1]
            Y = pl.linspace(years.min(),2017,2017-years.min()+1)
        elif years.min()<=1950 and years.max()<2017:
            y1 = pl.where(years==1950)[0][0]
            y2 = pl.where(years==years.max())[0][-1]
            Y = pl.linspace(1950,years.max(),years.max()-1950+1)
        elif years.min()>1950 and years.max()<2017:
            y1 = pl.where(years==years.min())[0][0]
            y2 = pl.where(years==years.max())[0][-1]
            Y = pl.linspace(years.min(),years.max(),years.max()-years.min()+1)
        
        fullyears = pl.linspace(1950,2017,68)
        Z = len(fullyears)-len(Y)
        monthly_means = pl.zeros([68,12]); monthly_means[:,:] = pl.float32('nan')
        
        if years.min()==years.max() and years.max()>2017:
            ta_o[s,:] = pl.float32('nan')
        else:
            for yr in range(len(Y)):
                #Z = pl.where(fullyears==Y[yr])[0][0]
                for mn in range(1,13):
                    #Y = years[y1:y2][yr]
                    #M = months[y1:y2][mn]
                    X = pl.where((years[y1:y2+1]==Y[yr]) & (months[y1:y2+1]==mn))[0]
                    monthly_means[yr+Z,mn-1] = pl.mean(temp[y1:y2+1][X[0]:X[-1]+1])
            
            ta_o[s,:] = pl.nanmean(monthly_means[:,3:10],axis=1)
    
    return ta_o, lats, lons, stations

def ERA5_data(dirpath,statlon,statlat):
    """Read the ERA5 t2m data & interpolate to station co-ordinates.
    """
    ncfile1 = xr.open_dataset(dirpath+'ERA5/era5_surf_mm_1950-1978.nc')
    era5lat = xr.DataArray(ncfile1.latitude)
    era5lon = xr.DataArray(ncfile1.longitude)
    t2m_early = xr.DataArray(ncfile1.t2m)
    ncfile1.close()
    
    ncfile2 = xr.open_dataset(dirpath+'ERA5/era5_surf_mm_1979_2017.nc')
    t2m_late = xr.DataArray(ncfile2.t2m)
    ncfile2.close()
    
    t2m = pl.zeros([t2m_early.shape[0]+t2m_late.shape[0],era5lat.size,
                                                        era5lon.size])
    
    t2m[:t2m_early.shape[0],:,:] = t2m_early.values
    t2m[t2m_early.shape[0]:,:,:] = t2m_late.values
    
    t2m = pl.reshape(t2m,newshape=(int(t2m.shape[0]/12),12,\
                                                    era5lat.size,era5lon.size))
    era5_tao = pl.mean(t2m[:,3:10,:,:],axis=1)
    
    era5_stats = pl.zeros([statlon.size,t2m.shape[0]])
    
    for i in range(era5_stats.shape[0]):
        point = (statlon[i],statlat[i])
        for j in range(era5_stats.shape[1]):
            era5_stats[i,j] = pc.BilinInterp(point,era5lon,era5lat,era5_tao[j]-273.15)
    
    return era5_tao, t2m, era5_stats, era5lat, era5lon

def INDECIS_data(dirpath,statlon,statlat):
    """Read the INDECIS gridded data & interpolate to station co-ordinates
    """
    ncfile = xr.open_dataset(dirpath+'ncfiles/ta_o_year.nc')
    indlat = xr.DataArray(ncfile.latitude)
    indlon = xr.DataArray(ncfile.longitude)
    ta_o = xr.DataArray(ncfile.ta_o)
    ncfile.close()
    
#    ta_o = pl.reshape(ta_o,newshape=(int(ta_o.shape[0]/12),12,\
#                                            indlon.size,indlat.size))
    ind_stats = pl.zeros([statlon.size,ta_o.shape[0]])
    
    for i in range(ind_stats.shape[0]):
        point = (statlon[i],statlat[i])
        for j in range(ind_stats.shape[1]):
            ind_stats[i,j] = pc.BilinInterp(point,indlon,indlat,ta_o[j].T)
    
    return ind_stats

def AreasCalc():
    """
    """
    glat = pl.arange(-89.875,89.876,0.25)
    glon = pl.arange(-179.875,179.876,0.25)
    
    # Convert lat & lon arrays to radians
    lat_rad = pl.radians(glat[:])
    lon_rad = pl.radians(pl.flipud(glon[:]))
    
    lat_half = pc.HalfGrid(lat_rad)
    nlon = lon_rad.size # number of longitude points
    delta_lambda = (2*pl.pi)/nlon


    #--------------calculate cell areas here, use function from above--------------
    # set up empty array for area, size lat_half X lon_half
    areas = pl.zeros([lon_rad.size,lat_rad.size])
    radius = 6.37*(10**6)
    # loop over latitude and longitude
    for i in range(glon.size): # loops over 256
        for j in range(lat_half.size-1): # loops over 512
            latpair = (lat_half[j+1],lat_half[j])
            areas[i,j] = pc.AreaFullGaussianGrid(radius,delta_lambda,latpair)
    
    areas_clip = areas[70:326,34:200]
    
    return areas_clip

def RegionCalc(vertices,lon2,lat2,data):
    """
    """
    rPath = mplPath.Path(vertices)
    TF = pl.zeros([lon2.size,lat2.size])
    rmask = pl.zeros([lon2.size,lat2.size])
    rmask[:] = pl.float32('nan')
    
    for i in range(lon2.size):
            for j in range(lat2.size):
                X = rPath.contains_point((lon2[i],lat2[j]))
                TF[i,j] = X
    
    Y = pl.where(TF)
    rmask[Y[0],Y[1]] = 1
    
    areas = AreasCalc()
    
    rdata = data[:,:,:]*rmask[:,:]#None,
    rareas = areas*rmask
    
    Q = pl.ones_like(data)
    f = pl.isnan(data)
    d = pl.where(f==True)
    Q[d[0],d[1],d[2]] = pl.float32('nan')
    
    #P = pl.average(rdata[0],weights=pl.nan_to_num(rareas))
    W = pl.zeros([data.shape[0]])
    W[0] = pl.float32('nan')
     
    for i in range(data.shape[0]): # loop over years
        W[i] = pl.nansum(rdata[i]*rareas)/(pl.nansum(rareas*Q[i]))
    
    return W

def Bulgaria():
    """
    """
    df = pd.read_csv(indecis+'permissions.txt')
    data = pl.asarray(df)
    
    PUB = 'non-public'
    
    bulgaria = pl.where((data[:,5]=='BULGARIA') & (data[:,2]=='tg') & (data[:,3]==PUB))[0]
    bulstats = data[bulgaria,4]
    bulstats = pl.asarray([i.strip() for i in bulstats])
    
    blats = data[bulgaria,6]/10; blons = pl.float32(data[bulgaria,7])/10
    
    return blats, blons

pl.close('all')

home = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
indecis = ncasdir + 'INDECIS/'

ncfile = xr.open_dataset(indecis+'ncfiles/ta_o_year.nc')
data = xr.DataArray(ncfile.variables['ta_o'][:])
lat = xr.DataArray(ncfile.variables['latitude'][:])
lon = xr.DataArray(ncfile.variables['longitude'][:])
ncfile.close()

orogfile = xr.open_dataset(ncasdir+'etopo05.nc')
orolat = xr.DataArray(orogfile.ETOPO05_Y)
orolon = xr.DataArray(orogfile.ETOPO05_X)
orog = xr.DataArray(orogfile.ROSE)
orogfile.close()
orolat = pl.flipud(orolat.values); orog = pl.flipud(orog.data)

countries = ['RO','RS','HU','UA','MD']
station_tao = []
#station_lats = []; station_lons = []

for i in range(len(countries)):
    if i == 0:
        nb_tao, nb_lats, nb_lons, stations = Nonblended(indecis,countries[i])
        station_tao = nb_tao
        station_lats.append(nb_lats)
        station_lons.append(nb_lons)
    elif i in (1,2):
        nb_tao, nb_lats, nb_lons, stations = Nonblended(indecis,countries[i])
        #station_tao.append(nb_tao)
        station_lats.append(nb_lats)
        station_lons.append(nb_lons)
    else:
        bl_tao, bl_lats, bl_lons, stations = Blended(indecis,countries[i])
        #station_tao.append(bl_tao)
        station_lats.append(bl_lats)
        station_lons.append(bl_lons)
        

ukraine = [(23.52,51.56),(25.73,52.02),(30.58,51.33),(30.95,52.12),(33.44,52.42),
           (35.65,50.45),(40.05,49.65),(39.72,47.80),(38.28,47.62),(38.23,47.14),
            (34.71,46.16),(36.55,45.38),(33.83,44.37),(32.71,45.51),(33.61,46.10),
            (31.86,46.28),(31.38,46.66),(29.66,45.41),(28.26,45.46),(29.17,46.41),
            (30.13,46.43),(29.20,48.04),(27.54,48.51),(25.11,47.82),(22.85,48.06),
             (22.13,48.44),(22.68,49.54),(24.07,50.55)]
romania = [(20.20,46.16),(21.17,46.41),(22.84,48.06),(24.95,47.79),(26.57,48.29),
           (28.23,46.62),(28.20,45.45),(29.67,45.21),(28.57,43.72),(22.59,44.21)]
hungary = [(16.18,46.91),(17.09,48.00),(18.73,47.87),(20.54,48.55),(22.14,48.44),
           (22.92,47.99),(22.03,47.61),(21.10,46.23),(19.60,46.17),(18.07,45.78)]
bulgaria = [(22.66,44.20),(23.03,43.80),(25.62,43.64),(27.02,44.17),(28.57,43.71),
            (27.97,42.03),(26.72,26.06),(26.12,41.34),(24.44,41.56),(22.93,41.32),
            (22.93,41.32),(22.32,42.31),(22.97,43.18),(22.31,43.81)]
serbia = [(18.88,45.91),(20.23,46.19),(22.14,44.51),(22.38,44.73),(22.70,44.48),
          (22.61,44.21),(22.37,43.79),(23.02,43.16),(22.38,42.34),(21.60,42.29),
        (21.76,22.70),(20.79,43.27),(20.32,42.86),(19.24,43.53),(19.11,44.50),
        (19.32,44.91),(19.14,44.95)]
moldova = [(26.60,48.26),(27.58,48.49),(29.11,47.98),(30.14,46.12),(29.20,46.55),
           (28.92,46.48),(28.97,46.05),(28.23,45.48),(28.22,46.66),(26.90,48.24)]

regions = [romania,hungary,bulgaria,serbia,ukraine,moldova]
labels = ['Romania','Hungary','Bulgaria','Serbia','Ukraine',
            'Moldova']
clist = ['b','darkgoldenrod','green','k','magenta','darkred']
#mrks = ['None','.','o','s','^','h']

x = pl.linspace(1,68,68)
years = pl.linspace(1950,2017,68).astype(int)
V = pl.zeros([len(regions),len(x)])
lines = []

lon2 = lon.values[70:326]
lat2 = lat.values[34:200]
data  = data.values[:,70:326,34:200]

for i in range(len(regions)):
    V[i] = RegionCalc(regions[i],lon2,lat2,data)


blats, blons = Bulgaria()

borders_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                                           '50m',edgecolor='grey',
                                        facecolor='none')

fig, ax = pl.subplots(1,3,figsize=(15,5))

ax1 = pl.subplot(131,projection=ccrs.PlateCarree(),extent=[19.5,30.5,42.5,49])
ax1.autoscale(enable=False)
ax1.coastlines(linewidth=0.5,resolution='50m')
ax1.add_feature(borders_50m,linewidth=0.5,zorder=5)
ax1.set_anchor('N')

clist = ['k']*station_tao.shape[0]
dflt_clrs = pl.rcParams['axes.prop_cycle'].by_key()['color']
clist[0] = dflt_clrs[0]
clist[1] = dflt_clrs[1]
clist[2] = dflt_clrs[2]
clist[3] = 'r'
clist[16] = 'r'
clist[23] = 'r'

ax1.pcolormesh(orolon.values,orolat,orog,cmap='Greys',norm=pl.Normalize(0,2000),
              alpha=0.7)
ax1.scatter(station_lons[0],station_lats[0],color=clist)
ax1.scatter(station_lons[1],station_lats[1],color='k',label='non-blended stations')
ax1.scatter(station_lons[2],station_lats[2],color='k')
ax1.scatter(station_lons[3],station_lats[3],color='k',marker='^',label='blended stations')
ax1.scatter(station_lons[4],station_lats[4],color='k',marker='^')
ax1.scatter(blons,blats,color='k',marker='x',label='non-public data')

ax1.legend(loc=(0.01,-0.5),scatterpoints=2,fontsize=13)
GridLines(ax1,False,True,True,False)
pl.title('a. ECA&D stations')

###############################################################################
ax2 = pl.subplot(132)
for i in range(station_tao.shape[0]):
    if i == 0:
        c = dflt_clrs[0]; zorder = 10; lw = 1.5
    elif i == 1:
        c = dflt_clrs[1]; zorder = 10; lw = 1.5
    elif i == 2:
        c = dflt_clrs[2]; zorder = 10; lw = 1.5
    elif i in (3,16,23):
        c = 'r'; zorder = 10; lw = 1.5
    else:
        c = 'silver'; zorder = 0; lw = 0.8
    ax2.plot(years,station_tao[i],color=c,zorder=zorder,lw=lw)
ax2.axvline(x=1961,ls='--',color='grey')
ax2.axvline(x=1993,ls='--',color='grey')
pl.xlim(1950,2017); pl.ylim(0,22)
pl.yticks([0,5,10,15,20])
ax2.grid(axis='y',ls='--',color='lightgrey')
pl.ylabel('$^\circ$C',fontsize=15,labelpad=-5)
pl.title('b. Romanian stations')

###############################################################################
ax3 = pl.subplot(133)
pl.xlim(1950,2017); pl.ylim(13,19)
ax3.grid(axis='y',ls='--',color='lightgrey')

clist = dflt_clrs[:len(regions)]#; lw
clist[0] = 'k'
for i in range(len(regions)):
    if i == 0:
        lw = 4; c = 'k'; zorder=10
    else:
        lw = 1.5; c = dflt_clrs[i-1]; zorder = 5
    ax3.plot(years,V[i,:],color=c,lw=lw,label=labels[i],zorder=zorder)
ax3.axvline(x=1961,ls='--',color='grey')
ax3.axvline(x=1993,ls='--',color='grey')
ax3.legend(loc=2,ncol=2,columnspacing=0.5,handletextpad=0.5)
pl.ylabel('$^\circ$C',fontsize=15,labelpad=-2)
pl.title('c. country area-averages')

pl.subplots_adjust(left=0.03,right=0.99,wspace=0.12)
#pl.tight_layout()

pl.savefig(indecis+'figures/stationdata_map_timeseries.png',dpi=600)