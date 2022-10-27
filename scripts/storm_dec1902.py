# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:19:07 2018

@author: np838619
"""

import pylab as pl
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
    p2 = pl.around(presobs,0); p2 = p2[~pl.isnan(presobs)].astype(int)
    crd_lon = coords[:,1][~pl.isnan(presobs)]
    crd_lat = coords[:,0][~pl.isnan(presobs)]
    texts = [pl.text(crd_lon[i],crd_lat[i],p2[i].astype(str)) for i in range(len(p2))]
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
        if ispddata[i][4:6] == month and ispddata[i][6:8] == "{0:0=2d}".format(DATESEL) and\
                                HOURSEL-3<=int(ispddata[i][8:10])<=HOURSEL+3\
                            and 40<=float(ispddata[i][35:40])<=75:
            if int(ispddata[i][8:10]) == HOURSEL+3 and 0<int(ispddata[i][10:12])<99:
                pass
            #elif int(ispddata[i][8:10]) == HOURSEL+3 and int(ispddata[i][10:12])==99:
            #    ispd_lons.append(float(ispddata[i][27:34]))
            #    ispd_lats.append(float(ispddata[i][35:40]))
            else:
                ispd_lons.append(float(ispddata[i][27:34]))
                ispd_lats.append(float(ispddata[i][35:40]))
    
    ispd_locs = pl.array([pl.asarray(ispd_lons),pl.asarray(ispd_lats)]).T
    
    axx.plot(ispd_locs[:,0],ispd_locs[:,1],marker='x',color='k',linewidth=0,
        transform=ccrs.PlateCarree(),alpha=0.25,ms=5)
    
    return None

def GridLines(ax,top,left,right,bottom):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
    gl.xlabels_top = top; gl.xlabels_bottom = bottom
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([-40,-30,-20,-10,0,10,20,30])
    gl.ylocator = mticker.FixedLocator([40,50,60,70,80])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':11}
    gl.ylabel_style = {'color': 'k','size':11}
    
    return None
    
pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
scoutdir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR_scout461/'
ispddir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/ISPD/'
year = '1902'
month = '12'

DATESEL = 26 # which day of the month?
HOURSEL = 18 # which hour of the day?
#ind = (DATESEL-1)*4 + HOURSEL/6 # which index? doesn't work for 3 hour forecasts

data = pl.genfromtxt(ncasdir+'latlon_files/latlon'+year+'_pc.txt')
morstats = pl.where((data[:,-4]==8) & (data[:,-3]==0)) # morning stations

evestats = pl.where((data[:,-2]==18) & (data[:,-1]==0)) # evening stations
evefrance = pl.where((data[:,-2]==17) & (data[:,-1]==51)) # French evening stats
eveall = pl.concatenate((evestats[0],evefrance[0]))

#allstats = pl.where((HOURSEL-3<data[:,-4]) & (data[:,-4]<HOURSEL+3))

#obs = pl.genfromtxt(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+month+'_27.csv',delimiter=',')

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
#pres[0] = pres[0]+0.04 # Sumburgh Head
#N = [1,2,14,15,16,17,18,19] # Malin Head/Shields and north (not Sumburgh Head)
#pres[N] = pres[N] + 0.03
#S = [2,3,4,5,6,7,8,9,10,13,14,20,21,22,23] # south of Malin/Shields
#pres[S] = pres[S] + 0.02
#pres[[11,12]] = pres[[11,12]] + 0.01 # Scilly & Jersey
#pres = pres*33.8639

pres[9] = pres[9] + 0.04 # Sumburgh Head
Nx = [10,11,24,25,26,27,28] # north of Malin Head & Shields
pres[Nx] = pres[Nx] + 0.03
# south of Malin Head & Shields, north of Jersey & Scilly
Sx = [12,13,14,15,16,17,18,19,22,23,29,30,31,32,33]
pres[Sx] = pres[Sx] + 0.02
pres[[20,21]] + 0.01 # Scilly & Jersey
pres = pres*33.8639


#oldfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu.nc')
#olddata = xr.DataArray(oldfile.PRMSL_P1_L101_GGA0)
#lat = xr.DataArray(oldfile.lat_0)
#lon = xr.DataArray(oldfile.lon_0)
#inittime = xr.DataArray(oldfile.initial_time0)
#foretime = xr.DataArray(oldfile.forecast_time0)
#
#oldmean = pl.mean(olddata,axis=0)
#oldsprd = pl.std(olddata,axis=0)

#newfile = xr.open_dataset(scoutdir+year+'/'+month+'/prmsl_eu.nc')
#newdata = xr.DataArray(newfile.PRMSL_P1_L101_GGA0)
#
#newmean = pl.mean(newdata,axis=0)
#newsprd = pl.std(newdata,axis=0)

#sprdfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu_sprd_'+year+month+'.nc')
#spread = xr.DataArray(sprdfile.PRMSL_sprd)

ind1, ind2, frc1, frc2, time = BetweenSteps(DATESEL,HOURSEL,inittime)

Mold = MakeAverage(oldmean,ind1,ind2,frc1,frc2,HOURSEL)/100
#Mnew = MakeAverage(newmean,ind1,ind2,frc1,frc2,HOURSEL)/100
Sold = MakeAverage(oldsprd,ind1,ind2,frc1,frc2,HOURSEL)/100
#Snew = MakeAverage(newsprd,ind1,ind2,frc1,frc2,HOURSEL)/100

#pl.figure(figsize=(12.5,8))
fig, ax = pl.subplots(1,2,figsize=(19,6.7))
pl.tight_layout()
pl.subplots_adjust(left=0.03,right=0.97,wspace=0.05,top=1.015,bottom=0.085)

ax1 = pl.subplot(121,projection=ccrs.PlateCarree())#pl.axes(projection=ccrs.PlateCarree())
ax1.set_extent([-10.001,30.001,49.999,75.001],ccrs.PlateCarree())
ax1.coastlines(color='grey',resolution='50m',linewidth=0.5)
ISPDstations(ax1,ispddir,year,month,DATESEL,HOURSEL)
X, Y = pl.meshgrid(lon,lat)
cn = ax1.contour(X,Y,Mold,norm=pl.Normalize(964,1060),
	colors='grey',levels=pl.linspace(964,1060,25),
                    alpha=0.5,transform=ccrs.PlateCarree())
#pl.colorbar(cs)

ax1.plot(coords[:,1],coords[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())
#obs_in = pl.delete(pres,[pres.size-6,pres.size-5,pres.size-4,pres.size-3,
#                         pres.size-2,pres.size-1])
#crd_in = pl.delete(coords,[coords.shape[0]-5,coords.shape[0]-4,coords.shape[0]-3,
#                           coords.shape[0]-1],axis=0)
obs_in = pres[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,
               24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42]]
crd_in = coords[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,
               24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42]]
PresLabels(obs_in,crd_in)
pl.clabel(cn,inline=True,fmt="%.0f",zorder=3,inline_spacing=5,manual=True)

land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
ax1.add_feature(land_50m,alpha=0.5)
#pl.title('(a) 20CRv3 ensemble mean & DWR observations',fontsize=10)
ax1.annotate('(a) 20CRv3 ensemble mean & DWR observations',(-9.5,74),fontsize=12,
             bbox={'facecolor':'w'})

ax2 = pl.subplot(122,projection=ccrs.PlateCarree())
ax2.set_extent([-10.001,30.001,49.999,75.001],ccrs.PlateCarree())
ax2.coastlines(color='grey',resolution='50m',linewidth=0.5)
cs = ax2.contourf(X,Y,Sold,norm=pl.Normalize(0,9),#colors='grey',
            levels=[1,1.5,2,2.5,3,3.5,4.5,5,6,7,8],alpha=0.4,cmap='OrRd',extend='both',
            transform=ccrs.PlateCarree())
#pl.colorbar(cs,extend='both',orientation='vertical')
#pl.clabel(cs,inline=True,fmt="%.1f",zorder=3,inline_spacing=5,manual=True)
ax2.plot(crd_in[:,1],crd_in[:,0],marker='.',color='k',linewidth=0,#alpha=0.25,
        transform=ccrs.PlateCarree())

errs = ErrorCalc(pres,Mold,Sold,coords,lon,lat)
#err_in = pl.delete(errs,[errs.size-5,errs.size-4,errs.size-3,errs.size-1])
err_in = errs[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,
               24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42]]
ErrLabels(err_in,crd_in,lon,lat)
#ax2.add_feature(land_50m,alpha=0.5)
#pl.title('(b) 20CRv3 ensemble spread & z-scores',fontsize=10)
ax2.annotate('(b) 20CRv3 ensemble spread & z-scores',(-9.5,74),fontsize=12,
             bbox={'facecolor':'w'})

GridLines(ax1,False,True,False,True)
GridLines(ax2,False,False,True,True)



f = pl.gcf()
colax = f.add_axes([0.6,0.07,0.3,0.03])                   
cb = pl.colorbar(cs,orientation='horizontal',cax=colax)
cb.set_ticks([1,1.5,2,2.5,3,3.5,4.5,5,6,7,8])
cb.set_label('hPa',fontsize=14)

#time = inittime.values[ind]
#time = list(time)
#time[0], time[3] = time[3], time[0]
#time[1], time[4] = time[4], time[1]
#time = ''.join(time)
pl.suptitle(time,y=0.99)

hour = time[-6:-4] + time[-3:-1]
day = time[:2]
savetime = year+month+day+'_'+hour
#pl.savefig(jashome+'/mslp_obs_'+year+'_comp/mslp_'+savetime+'_obs_sprd.png')


###############################################################################

#diff_mean = Mnew - Mold
#diff_sprd = Snew - Sold
#
#fig, ax = pl.subplots(1,2,figsize=(19,6.7))
#pl.tight_layout()
#pl.subplots_adjust(left=0.03,right=0.97,wspace=0.05,top=1.015,bottom=0.085)
#
#ax1 = pl.subplot(121,projection=ccrs.PlateCarree())
#ax1.set_extent([-20.001,20.001,39.999,65],ccrs.PlateCarree())
#ax1.coastlines(color='grey',resolution='50m',linewidth=0.5)
#
#cf1 = ax1.contourf(X,Y,diff_mean,transform=ccrs.PlateCarree(),alpha=0.4,
#                   cmap='seismic',norm=pl.Normalize(-2,2),extend='both',
#                    levels=[-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2])
#ax1.contour(X,Y,diff_mean,levels=[0],colors='w',linewidths=3)
#
#ax1.annotate('(e) change in ensemble mean',(-19.5,64.0),fontsize=12,
#             bbox={'facecolor':'w'})
#
#ax2 = pl.subplot(122,projection=ccrs.PlateCarree())
#ax2.set_extent([-20.001,20.001,39.999,65],ccrs.PlateCarree())
#ax2.coastlines(color='grey',resolution='50m',linewidth=0.5)
#
#cf2 = ax2.contourf(X,Y,diff_sprd,transform=ccrs.PlateCarree(),alpha=0.4,
#                   cmap='seismic',norm=pl.Normalize(-2,2),extend='both',
#                    levels=[-4,-2.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2])
#ax2.contour(X,Y,diff_sprd,levels=[0],colors='w',linewidths=3)
#
#ax2.annotate('(f) change in ensemble spread',(-19.5,64.0),fontsize=12,
#             bbox={'facecolor':'w'})
#
#GridLines(ax1,False,True,False,True)
#GridLines(ax2,False,False,True,True)
#
#f = pl.gcf()
#colax = f.add_axes([0.2,0.08,0.6,0.03])                   
#cb = pl.colorbar(cf1,orientation='horizontal',cax=colax)
#cb.set_ticks([-2,-1.5,-1,-0.5,-0.25,-0.1,0,0.1,0.25,0.5,1,1.5,2])
#cb.set_label('hPa',fontsize=14)

#pl.tight_layout()
#pl.subplots_adjust(left=0.04,right=0.96,top=1,bottom=0.12,wspace=0.1)

#pl.suptitle(time+' mslp scout minus 20CRv3',y=0.98)
#pl.savefig(jashome+'/mslp_obs_'+year+'_comp/mslp_'+savetime+'_diff_em_sprd_poster.png')

###############################################################################

#mean_sprd_diff = (pl.mean(newsprd,axis=(0,1)) - pl.mean(oldsprd,axis=(0,1)))/100
#prcnt_chng_a = (mean_sprd_diff/(pl.mean(oldsprd,axis=(0,1))/100))*100
#prcnt_chng_b = (pl.mean(newsprd-oldsprd,axis=(0,1))/(pl.mean(oldsprd,axis=(0,1))/100))

#fig, ax = pl.subplots(1,1,figsize=(8,5))
#
#ax1 = pl.subplot(111,projection=ccrs.PlateCarree())
#
#ax1.set_extent([-45.001,45.001,19.999,90],ccrs.PlateCarree())
#ax1.coastlines(color='grey',resolution='50m',linewidth=0.5)
#
#coords = data[:][:-1,1:3]
#ax1.plot(coords[:,1],coords[:,0],color='k',linewidth=0,marker='.',
#         transform=ccrs.PlateCarree())
#cf = ax1.contourf(X,Y,prcnt_chng_a,norm=pl.Normalize(-20,20),extend='both',
#                  alpha=0.4,cmap='seismic',transform=ccrs.PlateCarree(),
#                levels=[-20,-15,-10,-5,-2,0,2,5,10,15,20])
#                  #levels=[-0.4,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4])
#cb = pl.colorbar(cf)
##cb.set_ticks([-0.4,-0.3,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.3,0.4])
#cb.set_ticks([-20,-15,-10,-5,-2,0,2,5,10,15,20])
#cb.set_label('%',fontsize=14)
#ax1.contour(X,Y,prcnt_chng_a,levels=[0],colors='w',linewidths=3)
#
#gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
#                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
#                  zorder=3)
#gl.xlabels_top = False
#gl.ylabels_left = True; gl.ylabels_right = False
#gl.xlocator = mticker.FixedLocator([-45,-30,-15,0,15,30,45])
#gl.ylocator = mticker.FixedLocator([20,35,50,65,80,90])
#gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
#gl.xlabel_style = {'color': 'k','size':11}
#gl.ylabel_style = {'color': 'k','size':11}
#
#pl.title('Feb 1903 mslp ensemble spread: scout minus 20CRv3')
#pl.tight_layout()
#pl.subplots_adjust(right=1.0,left=0.00,bottom=0.05)
#pl.savefig(jashome+'/mslp_obs_'+year+'_comp/mslp_'+year+month+'_sprd_diff.png')
#pl.savefig(jashome+'/mslp_obs_'+year+'_comp/mslp_'+year+month+'_sprd_prcnt.png')