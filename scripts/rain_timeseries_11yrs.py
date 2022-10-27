# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:08:00 2018

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob
#from hf_func import OSGB36toWGS84
from hf_funcs import WGS84toOSGB36
from functions import NearestIndex
#from mpl_toolkits.basemap import Basemap
import sklearn.metrics as skm
import cartopy
import cartopy.crs as ccrs
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

years = pl.linspace(1900,1910,11).astype(int)

rainfiles = pl.zeros([len(years),12],dtype=object)

for yr in range(len(years)):
    rainfiles[yr] = glob.glob(raindir+'/rainfall_hadukgrid_uk_1km_day_'+
                                                    str(years[yr])+'*')

#rainfiles = glob.glob(raindir+'/rainfall_hadukgrid_uk_1km_day_1903*')
#rainfiles = pl.asarray(rainfiles)
#rainfiles = pl.sort(rainfiles)

dwrfiles = pl.zeros([len(years),366],dtype=object)

for yr in range(len(years)):
    files = glob.glob(ncasdir+'DWRcsv/'+str(years[yr])+'/*')
    if len(files) == 366:
        dwrfiles[yr] = pl.asarray(files)
    else:
        dwrfiles[yr,:-1] = pl.asarray(files)
#dwrfiles = glob.glob(ncasdir+'DWRcsv/1903/*')
#dwrfiles = pl.asarray(dwrfiles)
#dwrfiles = pl.sort(dwrfiles)

names = ['station','YE pres','YE temp','TM pres','TM dry','TM wet','P24 max',
         'P24 min', 'P24 rain'] # variable names from log book
#dwrinds = [9,10,16,17,18,19,20,22,23,24,25,26,27,28,29,30,32,33,56,57]
stations = ['Sumburgh Head ','Stornoway ','Wick ','Nairn ','Aberdeen ',
            'Leith ','Shields ','Spurn Head ','Donaghadee ','Liverpool ',
            'Holyhead ','Pembroke ','Scilly ','Portland Bill ',
            'Dungeness ','London ','Oxford ','Yarmouth ']
dwrrain = pl.zeros([len(years),366,len(stations)],dtype=object)
dwrinds = pl.zeros([len(years),len(stations)])

for yr in range(len(years)):
    df = pd.read_csv(dwrfiles[yr,0],header=None,names=names)
    logs = pl.array(df)
    S = logs[:,0]#.tolist()
    for i in range(len(stations)):
        w = pl.where(S==stations[i]); w = w[0][0]
        dwrinds[yr,i] = w#S.index(stations[i])
dwrinds = dwrinds.astype(int)

for yr in range(len(years)):
    if years[yr] in (1904,1908):
        for name in range(dwrfiles.shape[1]):
            df = pd.read_csv(dwrfiles[yr,name],header=None,names=names)
            logs = pl.array(df) # dataframe into array
            #space = pl.where(logs[:,-1])
            dwrrain[yr,name,:] = logs[:,-1][dwrinds[yr,:].astype(int)]
    else:
        for name in range(dwrfiles.shape[1]-1):
            df = pd.read_csv(dwrfiles[yr,name],header=None,names=names)
            logs = pl.array(df) # dataframe into array
            #space = pl.where(logs[:,-1])
            dwrrain[yr,name,:] = logs[:,-1][dwrinds[yr,:].astype(int)]

space = pl.where(dwrrain==' ')
space2 = pl.where(dwrrain=='  ')
dwrrain[space] = pl.float32('nan')
dwrrain[space2] = pl.float32('nan')
dwrrain = dwrrain.astype(float)
#dwrrain[:16] = pl.float32('nan')

dwrrain = pl.reshape(dwrrain,(dwrrain.shape[0]*dwrrain.shape[1],dwrrain.shape[2]))

delinds = pl.array([0,1,2,3,5,6,7,9,10])
#for i in range(len(delinds)):
dwrrain = pl.delete(dwrrain,365+366*delinds,axis=0)

#dwrstats = logs[dwrinds,0]

statlocs = pl.zeros([len(years),len(stations),2])
for yr in range(len(years)):
    for stat in range(len(stations)):
        data = pd.read_csv(ncasdir+'latlon_files/latlon'+str(years[yr])+'_pc.txt',
                                           header=None,delim_whitespace=True)
        data = pl.asarray(data)
        statlocs[yr,:,0] = data[dwrinds[yr],2] # lon
        statlocs[yr,:,1] = data[dwrinds[yr],1] # lat
#statlocs = pd.read_csv(jashome+'latlon1903_pc.txt',delim_whitespace=True)
#statlocs = pl.asarray(statlocs)
#
#sl_inds = [8,9,15,16,17,18,19,21,22,23,24,25,26,27,28,29,31,32,55,56]

#ncfile = Dataset(ncasdir+'lons_lats_1km_osgb_grid.nc','r')
#lons = ncfile.variables['longitudes'][:]
#lats = ncfile.variables['latitudes'][:]
#ncfile.close()
#
#rain = pl.zeros([rainfiles.shape[0],rainfiles.shape[1],31,lons.shape[1],lons.shape[0]])
#
#for yr in range(rainfiles.shape[0]):
#    for mn in range(rainfiles.shape[1]):
#        ncfile = Dataset(rainfiles[yr,mn],'r')
#        month = ncfile.variables['rainfall'][:]
#        ncfile.close()
#        if month.shape[0] < 31:
#            rain[yr,mn,:month.shape[0],:,:] = month[:,:,:]
#        else:
#            rain[yr,mn,:,:,:] = month[:,:,:]
#
#raindays = pl.reshape(rain,(11*12*31,1450,900))
#NZ = pl.count_nonzero(raindays,axis=(1,2))
## use pl.delete with indices from pl.where(NZ==0)
#WZ = pl.where(NZ==0)
#raindays = pl.delete(raindays,WZ[0],axis=0)
#
#x_coords = pl.arange(-200000+500,700001-500,1000)
#y_coords = pl.arange(-200000+500,1250001-500,1000)
#
#gridrain = pl.zeros_like(dwrrain)
#
#for stat in range(len(stations)):#
#    loc = (statlocs[0,stat,0],statlocs[0,stat,1]) # lon,lat
#    #print loc
##lonf = lons.flatten(); latf = lats.flatten()
##
##ind1 = NearestIndex(lonf,PB_loc[0]); ind2 = NearestIndex(latf,PB_loc[1])
#
#    a = WGS84toOSGB36(loc[1],loc[0])
#    ind1 = NearestIndex(x_coords,a[0]); ind2 = NearestIndex(y_coords,a[1])
#    if raindays[0,ind2,ind1] == raindays.max():
#        R = raindays[0,ind2-5:ind2+6,ind1-5:ind1+6]
#        N = pl.where(R!=R.max())
#        #I = NearestIndex(N[0],4)
#        #J = NearestIndex(N[1],4)
#        d = pl.sqrt((4-N[0][:])**2+(4-N[1][:])**2)
#        dm = pl.where(d==d.min())
#        i = N[0][dm[0][0]]; j = N[1][dm[0][0]]
#        latind = raindays.shape[1] - ind2 - (i-5)
#        lonind = ind1 - (5 - j)
#    else:
#        latind = raindays.shape[1]-ind2; lonind = ind1
#    
#    gridrain[:,stat] = raindays[:,-latind,lonind]
    #rmse = RMSE(dwrrain[1:,stat]*25.4,raindays[:-1,-latind,lonind])
    #mbe = MBE(dwrrain[1:,stat]*25.4,raindays[:-1,-latind,lonind])
    #print logs[dwrinds[stat],0], round(rmse,2), round(mbe,2)

#f = open(ncasdir+'dwrrain_18stations_1900-1910.csv','w')
#f.write(','.join(stations)+'\n')
##dwrrain.tofile(f,sep=",")#format=".2f"
#pl.savetxt(f,dwrrain,delimiter=',')
#f.close()
#
#f = open(ncasdir+'gridrain_18stations_1900-1910.csv','w')
#f.write(','.join(stations)+'\n')
#pl.savetxt(f,gridrain,delimiter=',')
#f.close()

#    if stat in (4,16):
#        ha = 'right'
#    else:
#        ha = 'left'
#    ax.plot(loc[0],loc[1],transform=ccrs.PlateCarree(),color='r',marker='o')
#    pl.text(loc[0]+0.2,loc[1]-0.15,dwrstats[stat][:-1],transform=ccrs.Geodetic(),
#            horizontalalignment=ha)
    #pl.figure(stat+1)
#    fig,ax = pl.subplots(1,2,figsize=(10,5))
#    
#    ax1 = pl.subplot(121)
#    ax1.plot(raindays[:,-latind,lonind],label='HadUK-Grid')
#    ax1.plot(dwrrain[:,stat]*25.4,label='DWR')
#    pl.grid(axis='y'); pl.xlim(0,365); pl.ylim(0,70); pl.legend()
#    pl.ylabel('mm',fontsize=15); pl.xlabel('days',fontsize=15)
#    pl.title('(a) '+dwrstats[stat]+'rainfall')
#    
#    ax2 = pl.subplot(122)
#    ax2.plot(dwrrain[1:,stat]*25.4-raindays[:-1,-latind,lonind])
#    pl.grid(axis='y')
#    pl.xlim(0,364); pl.ylim(-35,25)
#    pl.ylabel('mm',fontsize=15); pl.xlabel('days',fontsize=15)
#    pl.title('(b) DWR minus HadUK-Grid')
#    
#    #pl.title('Met Office minus Weather Rescue: '+logs[dwrinds[stat],0])
#    pl.tight_layout()
#    pl.savefig(jashome+'raincomp_plots_1km/rain_panels_1903_'+dwrstats[stat][:-1]+'.png')

#pl.tight_layout()