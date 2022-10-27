# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:02:40 2022

@author: pmcraig
"""


import pylab as pl
import glob
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import timeit
import matplotlib.ticker as mticker
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from adjustText import adjust_text
import cartopy.util as util
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

def ForecastSteps(DATESEL,HOURSEL,inititme,foretime):
    """
    """
    F = pl.array([foretime.values[0],foretime.values[1]],dtype='timedelta64[h]')
    ind = int((DATESEL-1)*4 + int(HOURSEL/6))
    frc = 1
    time = inittime.values[ind]
    time = list(time)
    time[0], time[3] = time[3], time[0]
    time[1], time[4] = time[4], time[1]
    HR = str("{0:0=2d}".format(int(time[-6:-4])+F[1].astype('int')))
    time[-6] = HR[0]; time[-5] = HR[1]
    time = ''.join(time)
    print time
    
    return ind, frc, time

def BetweenSteps(DATESEL,HOURSEL,inittime):
    """
    """
    H = int(HOURSEL/3)
    ind1 = int((DATESEL-1)*4 + int(HOURSEL/6))#; ind2 = (DATESEL-1)*4 + H+1
    if H % 2 == 0:
        ind2 = ind1
        frc1 = 0; frc2 = 1
    else:
        ind2 = ind1 + 1
        frc1 = 1; frc2 = 0
    time = inittime.values[ind1]
    time = list(time)
    time[0], time[3] = time[3], time[0]
    time[1], time[4] = time[4], time[1]
    HR = str("{0:0=2d}".format(HOURSEL))
    time[-6] = HR[0]; time[-5] = HR[1]
    time = ''.join(time)
    print time
    
    return ind1, ind2, frc1, frc2, time

def MakeAverage(data_in,ind1,ind2,frc1,frc2,HOURSEL):
    """
    """
    N = HOURSEL-int(HOURSEL/3)*3 # number of hours to muliply dp/dt below
    M = data_in[ind1,frc1]
    M2 = data_in[ind2,frc2]
    dp_dt = (M2-M)/(3*60*60)
    MI = M + dp_dt*(N*60*60)
    
    return MI

def PresLabels(presobs,coords):
    """
    Args:
        presobs (array): pressure observations
        coords (array): latitude, longitude co-ordinates of stations
    """
    #p2 = pl.around(presobs,0); p2 = p2[~pl.isnan(presobs)].astype(int)
    crd_lon = coords[:,1]#[~pl.isnan(presobs)]
    crd_lat = coords[:,0]#[~pl.isnan(presobs)]
    texts = [pl.text(crd_lon[i],crd_lat[i],str(presobs[i].astype(int))) for i in range(len(presobs))]
    adjust_text(texts,avoid_text=True,avoid_self=False)

    return None

def ErrorCalc(varobs,ensmean,spread,loc,lon,lat):
    """
    varmean (array): ensemble mean
    varsprd (array): ensemble spread
    locs (array): longitude, latitude co-ordinates of stations
    lon (array): longitude array from 20CR
    lat (array): latitude array from 20CR
    """
    errs = pl.zeros_like(varobs)
    for i in range(len(varobs)):
        #print i, varobs[i]
        if pl.isnan(varobs[i]) == True:
            errs[i] = pl.float32('nan')
        else:
            statmean = pc.BilinInterp((loc[i,1],loc[i,0]),lon,lat,ensmean)
            #print statmean.values
            statsprd = pc.BilinInterp((loc[i,1],loc[i,0]),lon,lat,spread)
            #print statsprd.values
            errs[i] = (statmean - varobs[i])/statsprd
    
    return errs

def ErrLabels(errors,coords,lon,lat):
    """
    """

    e2 = pl.around(errors,2); e2 = e2[~pl.isnan(errors)]
    crd_lon = coords[:,1][~pl.isnan(errors)]
    crd_lat = coords[:,0][~pl.isnan(errors)]
    texts = [pl.text(crd_lon[i],crd_lat[i],e2[i].astype(str)) for i in range(len(e2))]
    adjust_text(texts,avoid_text=True,avoid_self=False,avoid_points=True)
    
    return None

def ISPDstations(axx,ispddir,year,month,DATESEL,HOURSEL):
    """
    """
    ispdfile = ispddir + 'ISPD47_'+year+'.txt'
    with open(ispdfile,'r') as f:
        ispddata = f.readlines()
    
    ispd_lons = []; ispd_lats = []
    for i in range(len(ispddata)):
        if ispddata[i][4:6] == "{0:0=2d}".format(MONSEL) and\
                            ispddata[i][6:8] == "{0:0=2d}".format(DATESEL) and\
                                HOURSEL-3<=int(ispddata[i][8:10])<=HOURSEL+3\
                            and 40<=float(ispddata[i][35:40])<=65:
            if int(ispddata[i][8:10]) == HOURSEL+3 and int(ispddata[i][10:12])>0:
                pass
            else:
                ispd_lons.append(float(ispddata[i][27:34]))
                ispd_lats.append(float(ispddata[i][35:40]))
    
    ispd_locs = pl.array([pl.asarray(ispd_lons),pl.asarray(ispd_lats)]).T
    
    axx.plot(ispd_locs[:,0],ispd_locs[:,1],marker='x',color='k',linewidth=0,
        transform=ccrs.PlateCarree(),alpha=0.5,ms=6)
    
    return None

def GridLines(ax,top,left,right,bottom):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
    gl.xlabels_top = top; gl.xlabels_bottom = bottom
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([-40,-30,-20,-10,0,10,20,26,30])
    gl.ylocator = mticker.FixedLocator([30,40,50,60,70])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':9}
    gl.ylabel_style = {'color': 'k','size':9}
    
    return None

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/glosat/development/data/raw/20CRv3/451/subdaily/'
#scoutdir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR_scout461/'
ispddir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/ISPD/'
year = '1872'
#month = '01'

MONSEL = 1
DATESEL = 18 # which day of the month?
HOURSEL = 8 # which hour of the day?

###############################################################################
#statdata = pl.genfromtxt(ncasdir+'latlon_files/latlon'+year+'_pc.txt')
df1 = pd.read_csv(ncasdir+'latlon_files/latlon'+year+'_pc.txt',header=None,delimiter=' ')
statdata = pl.array(df1)
#allstats = pl.where((HOURSEL-3<data[:,4]) & (data[:,4]<HOURSEL+3))
#coords = statdata[:,1:3]

df2 = pd.read_csv(ncasdir+'DWRcsv/'+year+'/DWR_'+year+'_'+"{0:0=2d}".format(MONSEL)+\
                        '_'+"{0:0=2d}".format(DATESEL)+'.csv',header=None,skiprows=1)
logs = pl.array(df2)

#statinds = []
pvals = pl.where(logs[:,1]!=999.00)[0]
coords = pl.zeros([pvals.size,2])
presobs = logs[pvals,1].astype(float)#*33.8639

for i in range(pvals.size):
    statind = pl.where(statdata[:,0]==logs[pvals[i],0])[0]
    coords[i,0] = statdata[statind,1]
    coords[i,1] = statdata[statind,2]

#correct for gravity and change to hPa:
sumind = pl.where(logs[pvals,0]=='SUMBURGHHEAD')[0][0]
presobs[sumind] = presobs[sumind] + 0.04
#shieldsind = pl.where(logs[pvals,0]=='SHIELDS')[0][0]
norstats = ['THURSO','WICK','NAIRN','ABERDEEN','LEITH','ARDROSSAN',
            'GREENCASTLE','SHIELDS']
N = [pl.where(logs[pvals,0]==i)[0][0] for i in norstats]
#N = pl.where(coords[:,0]>=coords[shieldsind,0])[0]# Shields and north (not Sumburgh Head)
presobs[N] = presobs[N] + 0.03

soustats = ['SCARBOROUGH','LIVERPOOL','HOLYHEAD','YARMOUTH','VALENTIA',
            'ROCHESPOINT','LONDON','DOVER','PORTSMOUTH','PLYMOUTH']
S = [pl.where(logs[pvals,0]==i)[0][0] for i in soustats]

scillyind = pl.where(logs[pvals,0]=='SCILLY')[0][0]
#heldind = pl.where(logs[pvals,0]=='HELDER')[0][0]
#brusind = pl.where(logs[pvals,0]=='BRUSSELS')[0][0]
#cgnind = pl.where(logs[pvals,0]=='CAPGRISNEZ')[0][0]
#S = pl.where((coords[:,0]>coords[jerseyind,0]) & (coords[:,0]<coords[shieldsind,0]))[0]
#S = pl.delete(S,pl.where(S==heldind)[0][0])
presobs = presobs*33.8639

###############################################################################

allfiles = glob.glob(CR20dir+year+'/PRMSL*')

enslen = len(allfiles)
 # 256 lat, 512 lon
#extract range lon=(-45,45), lat=(20,80)
presdata = pl.zeros([enslen,128,512])

for nc in range(enslen):
    ncfile = xr.open_dataset(allfiles[nc])
    if nc == 0:
        lat = xr.DataArray(ncfile.lat[:128]).data
        lon = xr.DataArray(ncfile.lon).data
        time = xr.DataArray(ncfile.time).data
            
        timelist = [str(pd.to_datetime(i)) for i in time]
        month = pl.asarray([i[5:7] for i in timelist]).astype(int)
        day = pl.asarray([i[8:10] for i in timelist]).astype(int)
        hour = pl.asarray([i[11:13] for i in timelist]).astype(int)
    
    if HOURSEL % 3 == 0:
        if nc == 0:

            A = pl.where((month==MONSEL) & (day==DATESEL) & (hour==HOURSEL))
            ind = A[0][0]
    
        presdata[nc,:,:] = xr.DataArray(ncfile.PRMSL[ind,:128,:]).data
        ncfile.close()
    else:
        # need to interpolate between adjacent timesteps
        if nc == 0:
            H = hour[:8]
            NI = pc.NearestIndex(H,HOURSEL)
            if NI < HOURSEL:
                lower = H[NI]; upper = H[NI+1]
            else:
                lower = H[NI-1]; upper = H[NI]
            
            A = pl.where((month==MONSEL) & (day==DATESEL) & (hour==lower))
            B = pl.where((month==MONSEL) & (day==DATESEL) & (hour==upper))
            ind1 = A[0][0]; ind2 = B[0][0]
        
        # extract pres at timestep before HOURSEL
        P1 = xr.DataArray(ncfile.PRMSL[ind1,:128,:]).data
        # extract pres at timestep after HOURSEL
        P2 = xr.DataArray(ncfile.PRMSL[ind2,:128,:]).data
        ncfile.close()
        
        # interpolate
        N = HOURSEL - int(HOURSEL/3)*3
        dp_dt = (P2-P1)/(3*60*60)
        presdata[nc,:,:] = P1 + dp_dt*(N*60*60)

pres_mn = pl.mean(presdata,axis=0)
pres_sd = pl.std(presdata,axis=0)

lontemp = pl.zeros_like(lon)
lontemp[:lon.size/2] = lon[lon.size/2:] - 360
lontemp[lon.size/2:] = lon[:lon.size/2]
lon = lontemp.copy()
del lontemp

ptemp = pl.zeros_like(pres_mn)
ptemp[:,:lon.size/2] = pres_mn[:,lon.size/2:]
ptemp[:,lon.size/2:] = pres_mn[:,:lon.size/2]
pres_mn = ptemp.copy()
del ptemp

stemp = pl.zeros_like(pres_sd)
stemp[:,:lon.size/2] = pres_sd[:,lon.size/2:]
stemp[:,lon.size/2:] = pres_sd[:,:lon.size/2]
pres_sd = stemp.copy()
del stemp

###############################################################################

errs = ErrorCalc(presobs,pres_mn/100,pres_sd/100,coords,lon,lat)
#err_in = pl.delete(errs,[0,6,errs.size-5,errs.size-4,errs.size-3,errs.size-1])


###############################################################################
fig, ax = pl.subplots(1,2,figsize=(13.5,5))

ax1 = pl.subplot(121,projection=ccrs.PlateCarree(central_longitude=0))
ax1.set_extent([-20.001,15.001,37.999,65],ccrs.PlateCarree())
ax1.coastlines(color='grey',resolution='50m',linewidth=0.5)

cn = ax1.contour(lon,lat,pres_mn/100,norm=pl.Normalize(952,1060),colors='grey',
           levels=pl.linspace(952,1060,28),
                    alpha=0.5,transform=ccrs.PlateCarree())

#pl.clabel(cn,inline=True,fmt="%.0f",zorder=3,inline_spacing=5,manual=True)

ax1.plot(coords[:,1],coords[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())
PresLabels(pl.around(presobs,0),coords)
ISPDstations(ax1,ispddir,year,MONSEL,DATESEL,HOURSEL)

land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
ax1.add_feature(land_50m,alpha=0.5)
pl.title('20CRv3 ensemble mean mslp & DWR observations 18/01/1872 (08:00)',
         size=10)
#pl.tight_layout()
###############################################################################

#pl.figure(2)
ax2 = pl.subplot(122,projection=ccrs.PlateCarree(central_longitude=0))
ax2.set_extent([-20.001,15.001,37.999,65],ccrs.PlateCarree())
ax2.coastlines(color='grey',resolution='50m',linewidth=0.5)

cs = ax2.contourf(lon,lat,pres_sd/100,norm=pl.Normalize(0,12),cmap='OrRd',
                  levels=pl.linspace(0,12,13),extend='max',alpha=0.4,
            transform=ccrs.PlateCarree())
#cn2 = ax2.contour(lon,lat,pres_sd/100,transform=ccrs.PlateCarree())
#pl.clabel(cn2,inline=True,fmt="%.0f",zorder=3,inline_spacing=5,manual=True)

ax2.plot(coords[:,1],coords[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())

ErrLabels(errs,coords,lon,lat)
pl.title('20CRv3 ensemble spread & z-scores 18/01/1872 (08:00)',size=10)
#pl.colorbar(cs,orientation='horizontal')

GridLines(ax1,False,True,False,True)
GridLines(ax2,False,False,True,True)

ax1.text(-21.8,64,'a',size=14)
ax2.text(-21.3,64,'b',size=14)

f = pl.gcf()
colax = f.add_axes([0.945,0.07,0.02,0.85])
cb = pl.colorbar(cs,orientation='vertical',cax=colax)
cb.set_label('hPa',fontsize=12,labelpad=1)
cb.set_ticks(pl.linspace(0,12,13))
cb.set_ticklabels(pl.linspace(0,12,13).astype(int))
cb.ax.tick_params(labelsize=10)

pl.tight_layout()
pl.subplots_adjust(left=0.02,right=0.925,wspace=0.0,top=0.95,bottom=0.05)

savetime = year+"{0:0=2d}".format(MONSEL)+"{0:0=2d}".format(DATESEL) + \
                "{0:0=2d}".format(HOURSEL)

#pl.savefig(jashome+'/mslp_obs_'+year+'_comp/mslp_'+savetime+'_obs_sprd.png',dpi=400)
#pl.savefig(jashome+'/mslp_obs_'+year+'_comp/mslp_'+savetime+'_obs_sprd.pdf',dpi=400)