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
#import pandas as pd
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

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
raindir = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/day/v20181126/'
#CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
CR20dir = ncasdir + '20CR/'
year = '1907'
month = '01'

data = pl.genfromtxt(ncasdir+'latlon_files/latlon'+year+'_pc.txt')
evestats = pl.where(data[:,-4]==8)
#es1 = pl.where(data[:,-2]==18)
#es2 = pl.where(data[:,-2]==17)
#es1 = es1[0].tolist(); es1.remove(data.shape[0]-1)
#es1.extend(es2[0].tolist()); evestats = es1
evedata = data[evestats[0][:]]
#evedata = data[evestats]

#obs = pl.genfromtxt(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_01_12.csv',delimiter=',')
##evepres = obs[evestats[0][:-1],1]*33.8638
#evepres = obs[evestats[0][:-1],3]#*33.8638
##correct for gravity and change to hPa:
#evepres[0] = evepres[0]+0.04 # Sumburgh Head
#N = [1,2,15,16,17,18,19] # Malin Head/Shields and north (not Sumburgh Head)
#evepres[N] = evepres[N] + 0.03
#S = [3,4,5,6,7,8,9,10,13,14,20,21,22,23,24,25,26,27] # south of Malin/Shields
#evepres[S] = evepres[S] + 0.02
#evepres[[11,12]] = evepres[[11,12]] + 0.01 # Scilly & Jersey
#evepres = evepres*33.8638
#dwr_2802 = pl.genfromtxt(ncasdir+'DWRcsv/1903/dwr_1903_01_01.csv',delimiter=',')

#morn_stns = [43,44,45,46,47,48,49,50]
#even_stns = [9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
#             31,33,38,52,57]

#filenames = glob.glob(20CRdir+'1903/02/*') # all filenames in dir
#filenames = pl.asarray(filenames) # list into array
#filenames = pl.sort(filenames) # get filenames in correct order

#tempfile = 20CRdir + '1903/03/air.2m.nc'
#precfile = 20CRdir +'1903/03/prate.nc'
#start_time = timeit.default_timer()
#presfile = Dataset(CR20dir + year + '/12/prmsl_eu_em_'+year+'12.nc','r')
#mslp = presfile.variables['PRMSL_P1_L101_GGA0'][:]
#lat = presfile.variables['lat_0'][:]
#lon = presfile.variables['lon_0'][:]
#inittime = presfile.variables['initial_time0'][:]
#foretime = presfile.variables['forecast_time0'][:]
#presfile.close()
#elapsed = timeit.default_timer() - start_time
#print elapsed/60
meanfile = xr.open_dataset(CR20dir+year+'/'+month+'/prmsl_eu_em_'+year+month+'.nc')
ensmean = xr.DataArray(meanfile.PRMSL_P1_L101_GGA0)
lat = xr.DataArray(meanfile.lat_0)
lon = xr.DataArray(meanfile.lon_0)
inittime = xr.DataArray(meanfile.initial_time0)
foretime = xr.DataArray(meanfile.forecast_time0)

#precfile = Dataset(ncasdir + 'prate_eu_mean.nc','r')
#prec = precfile.variables['PRATE_P11_L1_GGA0_avg3h'][:]
#lon2 = precfile.variables['lon_0'][:]
#lat2 = precfile.variables['lat_0'][:]
#precfile.close()
#prec = prec[1:]

#ax = pl.axes(projection=ccrs.AlbersEqualArea(central_longitude=-2,
#				false_easting=400000,false_northing=-100000))
ax = pl.axes(projection=ccrs.PlateCarree())
ax.set_extent([-45,45,25,80],ccrs.PlateCarree())
ax.coastlines(color='grey',resolution='50m',linewidth=0.5)
X, Y = pl.meshgrid(lon,lat)
cs = pl.contour(X,Y,mslp[14,0]/100,norm=pl.Normalize(980,1020),
	colors='k',levels=pl.linspace(980,1020,11),extend='max',
                    transform=ccrs.PlateCarree())
pl.clabel(cs,inline=True,fmt="%.0f",zorder=3)

#for i in enumerate(pl.linspace(104,116,12)):#104,116,12
#    for frc in range(0,2):
#        ind = int(i[1])#60 # which index in zero-axis of mslp_em to use
#        #frc = 0 # 0 or 1, which forecast time to use (+ 0 hours or + 3 hours)
#        #inittime[ind,-5] = str(int(inittime[ind,-5])+foretime[frc])
#        hour = str("{0:0=2d}".format(int(''.join(inittime[ind,-6:-4]))+foretime[frc]))
#        inittime[ind,-6] = hour[0]; inittime[ind,-5] = hour[1]
#        time = ''.join([str(x) for x in inittime[ind]]) # combine strings into one
#        print time
#    
#        pl.figure(figsize=(13,10))#2*i[0]+frc,
#        m = Basemap(projection='cyl',llcrnrlon=-20,urcrnrlon=25,llcrnrlat=40,
#                    urcrnrlat=70,resolution='l')
#        m.drawcoastlines(color='grey',linewidth=0.5)
#    
#        M = mslp_em[ind,frc]/100
#        #M, lonx = shiftgrid(180.0, mslp_em[ind]/100, lon, start=False)
#        P = prec[ind,frc]*10800
#        #P, lonx = shiftgrid(180.0,prec[ind,frc]*10800,lon,start=False)
#        
#        #M2 = mslp_em[ind,1]/100 
#        #dp_dt = (M2-M)/(3*60*60)
#        #MI = M + dp_dt*(2*60*60)
#        lons,lats = pl.meshgrid(lon,lat)
#        X, Y = m(lons,lats)
#        levels = pl.arange(940,1040,4)
#        ct = m.contour(X,Y,M,levels=levels,zorder=2)
#        cp = m.pcolormesh(X,Y,P,norm=pl.Normalize(0,4),cmap='Blues')
#        pl.clabel(ct,inline=True,fmt="%.0f",zorder=3)
#        cb = m.colorbar(cp,extend='max')
#        cb.set_label('mm')

#m.plot(evedata[:,2],evedata[:,1],latlon=True,marker='.',color='k',linewidth=0)
#for i in range(evepres.size):
#    if pl.isnan(evepres[i]) == True:
#        pass
#    elif i == 14:
#        pl.annotate(str(int(round(evepres[i],0))),(evedata[i,2],evedata[i,1]),
#                    zorder=4,textcoords='offset pixels',xytext=(-40,-6))
#    elif i == 23:
#        pl.annotate(str(int(round(evepres[i],0))),(evedata[i,2],evedata[i,1]),
#                    zorder=4,textcoords='offset pixels',xytext=(-15,4))
#    else:
#		pl.annotate(str(int(round(evepres[i],0))),(evedata[i,2],evedata[i,1]),
#                    zorder=4,textcoords='offset pixels',xytext=(2,-2))

#        m.drawmeridians([-20,-10,0,10,20],linewidth=0.5,dashes=[1,2],zorder=3,
#                        color='grey',labels=[1,1,0,1],fontsize=15)
#        m.drawparallels([40,50,60,70],linewidth=0.5,dashes=[1,2],zorder=3,color='grey',
#                        labels=[1,0,0,0],fontsize=15)
#        
#        pl.title('mslp '+time)
#        pl.tight_layout()
#        pl.subplots_adjust(bottom=0.05,left=0.06,right=0.94)
#        year = time[6:10]; month = time[0:2]; day = time[3:5]
#        step = time[-6:-4]+time[-3:-1]
#        savetime = year+month+day+'_'+step#+'_'
        #pl.savefig(jashome+'mslp_'+year+'_maps/'+month+'/mslp_'+savetime+'.png')
#pl.show()

#loc = (-1.44,55.03)
#stpr = pc.BilinInterp(loc,lon,lat,mslp_em[ind,frc])
#print stpr

#print prec[104:112,0].max()
#
#fig, ax = pl.subplots(2,4,figsize=(17,7))
#
#levels = pl.arange(940,1040,4)
#lons,lats = pl.meshgrid(lon,lat)
#m = Basemap(projection='cyl',llcrnrlon=-20,urcrnrlon=15,llcrnrlat=40,
#                    urcrnrlat=65,resolution='l')
#X, Y = m(lons,lats)
#
#frc = 0
#oct27 = [104,105,106,107]
#titles = ['(a) ','(b) ','(c) ','(d) ']
#for i in range(len(oct27)):
#    axx = pl.subplot(2,4,i+1)
#    ind = oct27[i]
#    M = mslp[ind,frc]/100
#    P = prec[ind,frc]*(10**4)
#    m.drawcoastlines(color='grey',linewidth=0.5)
#    ct = m.contour(X,Y,M,zorder=2,levels=levels,colors='k',linewidths=0.5)#
#    cp = m.pcolormesh(X,Y,P,norm=pl.Normalize(0,4),cmap='Blues')#
#    pl.clabel(ct,inline=True,fmt="%.0f",zorder=3)
#    m.drawmeridians([-20,-10,0,10],linewidth=0.5,dashes=[1,2],zorder=3,
#                        color='grey',labels=[0,0,0,0],fontsize=10)
#    if i == 0:
#        l = 1
#    else:
#        l = 0
#    m.drawparallels([40,50,60],linewidth=0.5,dashes=[1,2],zorder=3,
#                    color='grey',labels=[l,0,0,0],fontsize=10)
#    time = MakeTime(inittime,foretime,ind,frc)
#    pl.title(titles[i]+time,loc='center')
#
#oct28 = [108,109,110,111]
#titles = ['(e) ','(f) ','(g) ','(h) ']
#for i in range(len(oct28)):
#    axx = pl.subplot(2,4,i+5)
#    ind = oct28[i]
#    M = mslp[ind,frc]/100
#    P = prec[ind,frc]*(10**4)
#    m.drawcoastlines(color='grey',linewidth=0.5)
#    ct = m.contour(X,Y,M,zorder=2,levels=levels,colors='k',linewidths=0.5)#
#    cp = m.pcolormesh(X,Y,P,norm=pl.Normalize(0,4),cmap='Blues')#
#    pl.clabel(ct,inline=True,fmt="%.0f",zorder=3)
#    m.drawmeridians([-20,-10,0,10],linewidth=0.5,dashes=[1,2],zorder=3,
#                        color='grey',labels=[0,0,0,1],fontsize=10,yoffset=1)
#    if i == 0:
#        l = 1
#    else:
#        l = 0
#    m.drawparallels([40,50,60],linewidth=0.5,dashes=[1,2],zorder=3,
#                    color='grey',labels=[l,0,0,0],fontsize=10)
#    time = MakeTime(inittime,foretime,ind,frc)
#    pl.title(titles[i]+time,loc='center')
#
#f = pl.gcf()
#colax = f.add_axes([0.945,0.12,0.015,0.75])
#cb = fig.colorbar(cp,extend='max',orientation='vertical',cax=colax)
#cb.set_label('kg m$^{-2}$ s$^{-1}$',size=15)
#
#pl.subplots_adjust(left=0.03,right=0.93,hspace=-0.04,wspace=0.07,
#                   top=0.96,bottom=0.03)
##pl.tight_layout()
#pl.show()
