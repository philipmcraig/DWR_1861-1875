# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:08:00 2018

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
#from hf_func import OSGB36toWGS84
from hf_funcs import WGS84toOSGB36, OSGB36toWGS84
from functions import NearestIndex
from mpl_toolkits.basemap import Basemap
import sklearn.metrics as skm
from netCDF4 import Dataset

def RMSE(obs,mod):
    """Calculate root mean squared error and take nan values into account.
    
    Args:
        obs (array): observed data
        mod (array): predicted/modelled data
    
    Returns:
        rmse (float): root mean square error of data
    """
    z = pl.where(pl.isnan(obs)==True)
    obs = pl.delete(obs,z)
    mod = pl.delete(mod,z)
    
    #rmse = pl.sqrt(skm.mean_squared_error(obs,mod))
    rmse = pl.sqrt(((mod-obs)**2).mean())
    
    return rmse

def MBE(obs,mod):
    """Calculate mean bias error and take nan values into account.
    
    Args:
        obs (array): observed data
        mod (array): predicted/modelled data
    
    Returns:
        mbe (float): mean bias error of data
    """
    z = pl.where(pl.isnan(obs)==True)
    obs = pl.delete(obs,z)
    mod = pl.delete(mod,z)
    
    mbe = (obs-mod).mean()
    
    return mbe

pl.close('all')
jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
raindir = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/day/v20181126'

rainfiles = glob.glob(raindir+'/rainfall_hadukgrid_uk_1km_day_1903*')
rainfiles = pl.asarray(rainfiles)
rainfiles = pl.sort(rainfiles)

dwrfiles = glob.glob(ncasdir+'DWRcsv/1903/*')
dwrfiles = pl.asarray(dwrfiles)
dwrfiles = pl.sort(dwrfiles)

names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book
dwrinds = [9,10,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33]
#dwrinds = list(pl.asarray(dwrinds)+1)
dwrrain = pl.zeros([dwrfiles.size,len(dwrinds)-1],dtype=object)

for name in range(dwrfiles.size):
    df = pd.read_csv(dwrfiles[name],header=None,names=names)
    logs = pl.array(df) # dataframe into array
    #     #space = pl.where(logs[:,-1])
    #dwrrain[name] = logs[:,-1][dwrinds[:-1]]
    k = logs[:,-1][dwrinds[:-1]]
    
    space = pl.where(k==' ')#pl.where(dwrrain==' ')
    space2 = pl.where(k=='  ')#pl.where(dwrrain=='  ')
    k[space] = pl.float32('nan')#dwrrain[space] = pl.float32('nan')
    k[space2] = pl.float32('nan')#dwrrain[space2] = pl.float32('nan')
    
    dwrrain[name] = k#logs[:,-1][dwrinds[:-1]]
    
    dwrrain = dwrrain.astype(float)
     #dwrrain[:16] = pl.float32('nan')

    if name == 0:
        dwrstats = logs[dwrinds[:-1],0]

statlocs = pd.read_csv(ncasdir+'latlon_files/latlon1903_pc.txt',delim_whitespace=True,
                       header=None)
statlocs = pl.asarray(statlocs)

sl_inds = [9,10,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31,32,33]#[9,10,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]#,55,56]

ncfile = Dataset(ncasdir+'lons_lats_1km_osgb_grid.nc','r')
lons = ncfile.variables['longitudes'][:]
lats = ncfile.variables['latitudes'][:]
ncfile.close()

rain = pl.zeros([len(rainfiles),31,lons.shape[1],lons.shape[0]])

for i in range(len(rainfiles)):
    ncfile = Dataset(rainfiles[i],'r')
    month = ncfile.variables['rainfall'][:]
    ncfile.close()
    if month.shape[0] < 31:
        rain[i,:month.shape[0],:,:] = month[:,:,:]
    else:
        rain[i,:,:,:] = month[:,:,:]
#
raindays = pl.reshape(rain,(12*31,1450,900))
NZ = pl.count_nonzero(raindays,axis=(1,2))
# use pl.delete with indices from pl.where(NZ==0)
WZ = pl.where(NZ==0)
raindays = pl.delete(raindays,WZ[0],axis=0)

x_coords = pl.arange(-200000+500,700001-500,1000)
y_coords = pl.arange(-200000+500,1250001-500,1000)

#lonlat = pl.zeros([x_coords.size,y_coords.size],dtype=object)
#lons = pl.zeros([x_coords.size,y_coords.size])
#lats = pl.zeros([x_coords.size,y_coords.size])
#
#for i in range(x_coords.size):
#    for j in range(y_coords.size):
#        a,b = OSGB36toWGS84(x_coords[i],y_coords[j])
#        #lonlat[i,j] = (a,b)
#        lons[i,j] = a; lats[i,j] = b

llfile = Dataset(ncasdir+'lons_lats_1km_osgb_grid.nc')
lons = llfile.variables['longitudes'][:]
lats = llfile.variables['latitudes'][:]
llfile.close()

#ex_names = [r'\textit{Southport}',r'\textit{Aberystywth}',r'\textit{Plymouth}',
#            r'\textit{Margate}',r'\textit{Lowestoft}',r'\textit{Harrogate}',
#            r'\textit{Manchester}',r'\textit{Darwen}',r'\textit{Birmingham}']
#ex_locs = pl.array([[53.64,-3.01],
#					[52.42,-4.08],
#					[50.37,-4.14],
#					[51.39,1.38],
#					[52.48,1.75],
#					[53.99,-1.54],
#					[53.48,-2.24],
#					[53.70,-2.49],
#					[52.48,-1.90]])
#
#pl.figure(figsize=(8.5,9.5))
#m = Basemap(projection='tmerc',llcrnrlon=-9,llcrnrlat=49,urcrnrlat=61,urcrnrlon=6,
#			lon_0=-1.5,lat_0=55,resolution='i')
#m.drawcoastlines(linewidth=0.5,color='lightgrey')
#m.drawcountries()
#m.plot(statlocs[sl_inds,2],statlocs[sl_inds,1],latlon=True,color='r',lw=0,
#       ms=10,marker='.',label='Page 1 stations')
#for i in range(len(sl_inds)):
#    x = statlocs[sl_inds[i],2]; y = statlocs[sl_inds[i],1]
#    x,y = m(x,y)
#    if i == 3:
#        pl.annotate(logs[:,0][sl_inds[i]+1],(x,y),xycoords='data',xytext=(-50,-15),
#                textcoords='offset pixels')
#    elif i == 4:
#        pl.annotate(logs[:,0][sl_inds[i]+1],(x,y),xycoords='data',xytext=(-60,-5),
#                textcoords='offset pixels')
#    elif i == 16:
#    	pl.annotate(logs[:,0][sl_inds[i]+1],(x,y),xycoords='data',xytext=(-20,-15),
#                textcoords='offset pixels')
#    else:
#        pl.annotate(logs[:,0][sl_inds[i]+1],(x,y),xycoords='data',xytext=(4,0),
#                textcoords='offset pixels')
#
#pl.matplotlib.rc('text', usetex=True)
#
#m.plot(ex_locs[:,1],ex_locs[:,0],latlon=True,color='b',lw=0,ms=10,marker='.',
#       label=r'\textit{Page 4 stations}')
#for i in range(len(ex_names)):
#	x = ex_locs[i,1]; y = ex_locs[i,0]
#	x,y = m(x,y)
#	if i == 0:
#		pl.annotate(ex_names[i],(x,y),xycoords='data',fontstyle='italic',xytext=(-65,-5),
#					textcoords='offset pixels')
#	else:
#		pl.annotate(ex_names[i],(x,y),xycoords='data',fontstyle='italic',xytext=(5,-5),
#					textcoords='offset pixels')
#
#leg = pl.legend(loc=2,fontsize=13,handletextpad=-0.5)
#pl.tight_layout()
#pl.show()

for stat in range(17,18):#len(sl_inds)
    loc = (statlocs[sl_inds[stat],2],statlocs[sl_inds[stat],1]) # lon,lat
    #print loc
#lonf = lons.flatten(); latf = lats.flatten()
#
#ind1 = NearestIndex(lonf,PB_loc[0]); ind2 = NearestIndex(latf,PB_loc[1])

    a = WGS84toOSGB36(loc[1],loc[0])
    ind1 = NearestIndex(x_coords,a[0]); ind2 = NearestIndex(y_coords,a[1])
    if raindays[0,ind2,ind1] == raindays.max():
        R = raindays[0,ind2-5:ind2+6,ind1-5:ind1+6]
        N = pl.where(R!=R.max())
        #I = NearestIndex(N[0],4)
        #J = NearestIndex(N[1],4)
        d = pl.sqrt((4-N[0][:])**2+(4-N[1][:])**2)
        dm = pl.where(d==d.min())
        i = N[0][dm[0][0]]; j = N[1][dm[0][0]]
        latind = raindays.shape[1] - ind2 - (i-5)
        lonind = ind1 - (5 - j)
    else:
        latind = raindays.shape[1]-ind2; lonind = ind1
#    
#    rmse = RMSE(dwrrain[1:,stat]*25.4,raindays[:-1,-latind,lonind])
#    mbe = MBE(dwrrain[1:,stat]*25.4,raindays[:-1,-latind,lonind])
#    print logs[dwrinds[stat],0], round(rmse,2), round(mbe,2)
#    
##    if stat in (4,16):
##        ha = 'right'
##    else:
##        ha = 'left'
##    ax.plot(loc[0],loc[1],transform=ccrs.PlateCarree(),color='r',marker='o')
##    pl.text(loc[0]+0.2,loc[1]-0.15,dwrstats[stat][:-1],transform=ccrs.Geodetic(),
##            horizontalalignment=ha)
#    #pl.figure(stat+1)
    fig,ax = pl.subplots(1,2,figsize=(10,5))
    
    ax1 = pl.subplot(121)
    ax1.plot(raindays[:,-latind,lonind],label='HadUK-Grid')
    ax1.plot(dwrrain[:,stat]*25.4,label='DWR')
    pl.grid(axis='y'); pl.xlim(0,365); pl.ylim(0,70)
    pl.legend(fontsize=14,framealpha=1.0)
    pl.ylabel('mm',fontsize=15); pl.xlabel('days',fontsize=15)
    pl.title('(a) '+dwrstats[stat]+'rainfall')
    
    ax2 = pl.subplot(122)
    ax2.plot(dwrrain[1:,stat]*25.4-raindays[:-1,-latind,lonind])
    pl.grid(axis='y')
    pl.xlim(0,364); pl.ylim(-35,25)
    pl.ylabel('mm',fontsize=15); pl.xlabel('days',fontsize=15)
    pl.title('(b) HadUK-Grid minus DWR')
    
    pl.tight_layout()
    pl.savefig(jashome+'raincomp_plots_1km/rain_panels_1903_'+dwrstats[stat][:-1]+'.png')

#pl.tight_layout()