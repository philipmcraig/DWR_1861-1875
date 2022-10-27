# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:35:32 2019

@author: pmcraig
"""

from __future__ import division
import pylab as pl
import glob
import xarray as xr
import pandas as pd
import pcraig_funcs as pc

def MakeTimeSeries(month1,month2):
    """
    """
    month1 = pl.asarray(month1)
    month2 = pl.asarray(month2)
    
    M1 = month1[16*4:]; M2 = month2[:16*4]
    flat1 = pl.reshape(M1,newshape=(M1.shape[0]*M1.shape[1],
                                    M1.shape[2],M1.shape[3]))
    flat2 = pl.reshape(M2,newshape=(M2.shape[0]*M2.shape[1],
                                    M2.shape[2],M2.shape[3]))
    
    M1S = pl.zeros_like(flat1); M2S = pl.zeros_like(flat1)
    
    M1S[:-4,:,:] = flat1[4:,:,:]
    M1S[-4:,:,:] = flat2[:4,:,:]
    M2S[:,:,:] = flat2[4:-4,:,:]
    
    return M1S, M2S

def DWRTemps(dwrfiles,loc):
    """
    """
    dwrtemps = pl.zeros([len(dwrfiles)])
    for i in range(len(dwrfiles)):
        logs = pd.read_csv(dwrfiles[i],header=None)
        obs = pl.array(logs)
        ind = pl.where(obs[:,0]==loc+' '); ind = ind[0]
        if obs[ind,-3] == ' ':
             dwrtemps[i] = pl.float32('nan')
        else:
             dwrtemps[i] = pl.float32(obs[ind,-3])
    
    dwrtemps = (dwrtemps-32)*(5./9.)
    
    return dwrtemps

def TempAndSpread(temps,augsprd,sepsprd,lat,lon,coords):
    """
    """
    Tstat = pl.zeros([temps.shape[0],temps.shape[1]])
    
    sprds = pl.concatenate((augsprd,sepsprd),axis=0)
    sprds = pl.reshape(sprds,newshape=(30,8,lat.size,lon.size))
    Sstat = pl.zeros_like(Tstat)
    
    Tmax = pl.zeros([Tstat.shape[0]]); Smax = pl.zeros_like(Tmax)
    
    for i in range(temps.shape[0]):
        for j in range(temps.shape[1]):
            Tstat[i,j] = pc.BilinInterp(coords,lon,lat,temps[i,j,:,:]) - 273.15
            Sstat[i,j] = pc.BilinInterp(coords,lon,lat,sprds[i,j,:,:])
    
        ind = pl.where(Tstat[i]==Tstat[i].max()); ind = ind[0][0]
        Tmax[i] = Tstat[i].max()
        Smax[i] = Sstat[i,ind]

    return Tmax, Smax

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
#CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
CR20dir = ncasdir + '20CR/'
year = '1906'
months = ['08','09']

coords = [(-3.8955,57.577615),(-1.154045,52.949697),(2.8,48.8),(13.4,52.5)]
locs = ['Nairn','Nottingham','Paris','Berlin']

dwraug = glob.glob(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+months[0]+'*')
dwrsep = glob.glob(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+months[1]+'*')

dwraug = dwraug[17:]; dwrsep = dwrsep[:16]
dwrfiles = dwraug + dwrsep
dwrtemps = pl.zeros([4,len(dwrfiles)])

#for i in range(len(dwrfiles)):
#    logs = pd.read_csv(dwrfiles[i],header=None)
#    obs = pl.array(logs)
#    ind = pl.where(obs[:,0]==loc+' '); ind = ind[0]
#    if obs[ind,-3] == ' ':
#         dwrtemps[i] = pl.float32('nan')
#    else:
#         dwrtemps[i] = pl.float32(obs[ind,-3])
#
#dwrtemps = (dwrtemps-32)*(5./9.)
for L in range(len(locs)):
    dwrtemps[L] = DWRTemps(dwrfiles,locs[L])

augfile = xr.open_dataset(CR20dir+year+'/'+months[0]+'/'+'air.2m_eu_190608.nc')
augdata = xr.DataArray(augfile.TMP_P1_L103_GGA0)
lon = xr.DataArray(augfile.lon_0)
lat = xr.DataArray(augfile.lat_0)

augfile = xr.open_dataset(CR20dir+year+'/'+months[0]+'/'+'air.2m_eu_sprd_190608.nc')
augsprd = xr.DataArray(augfile.AIR2M_sprd)

sepfile = xr.open_dataset(CR20dir+year+'/'+months[1]+'/'+'air.2m_eu_em_190609.nc')
sepdata = xr.DataArray(sepfile.TMP_P1_L103_GGA0)

sepfile = xr.open_dataset(CR20dir+year+'/'+months[1]+'/'+'air.2m_eu_sprd_190609.nc')
sepsprd = xr.DataArray(sepfile.AIR2M_sprd)

augtemps, septemps = MakeTimeSeries(augdata,sepdata)
augsprd, sepsprd = MakeTimeSeries(augsprd,sepsprd)

temps = pl.concatenate((augtemps,septemps),axis=0)
temps = pl.reshape(temps,newshape=(30,8,
                                   lat.size,lon.size))

Tmax = pl.zeros([4,temps.shape[0]]); Smax = pl.zeros_like(Tmax)

for C in range(len(coords)):
    Tmax[C], Smax[C] = TempAndSpread(temps,augsprd,sepsprd,lat,lon,coords[C])
#Tmax = pl.amax(temps,axis=1)
#Tstat = pl.zeros([temps.shape[0],temps.shape[1]])
#
#sprds = pl.concatenate((augsprd,sepsprd),axis=0)
#sprds = pl.reshape(sprds,newshape=(30,8,
#                                   lat.size,lon.size))
#Sstat = pl.zeros_like(Tstat)
#
#Tmax = pl.zeros([Tstat.shape[0]]); Smax = pl.zeros_like(Tmax)
#
#for i in range(temps.shape[0]):
#    for j in range(temps.shape[1]):
#        Tstat[i,j] = pc.BilinInterp(coords,lon,lat,temps[i,j,:,:]) - 273.15
#        Sstat[i,j] = pc.BilinInterp(coords,lon,lat,sprds[i,j,:,:])
#
#    ind = pl.where(Tstat[i]==Tstat[i].max()); ind = ind[0][0]
#    Tmax[i] = Tstat[i].max()
#    Smax[i] = Sstat[i,ind]

labels = ['(a)','(b)','(c)','(d)']
xticklabs = ['Aug 17','Aug 21','Aug 26','Aug 31','Sep 5','Sep 10','Sep 15']
fig, ax = pl.subplots(2,2,figsize=(9,7))
for i in range(4):
    axx = pl.subplot(2,2,i+1)
    axx.plot(pl.linspace(0,len(Tmax[i])-1,len(Tmax[i])),Tmax[i],color='k',lw=2,
            label='20CRv3')
    axx.plot(pl.linspace(0,len(Tmax[i])-1,len(Tmax[i])),dwrtemps[i],
             color='orange',lw=2,ls='-',label=' DWR')
    pl.fill_between(pl.linspace(0,len(Tmax[i])-1,len(Tmax[i])),Tmax[i]-Smax[i],
                                                Tmax[i]+Smax[i],alpha=0.2)
    pl.ylim(10,35); pl.xlim(0,len(Tmax[i])-1)
    pl.grid(axis='y',ls='--')
    pl.ylabel('$^\circ$C',fontsize=16)
    pl.xticks([0,4,9,14,19,24,29])
    axx.set_xticklabels(xticklabs)
    axx.tick_params(axis='y',direction='in')
    axx.tick_params(axis='x',pad=7,direction='in',length=3,top=True)
    pl.title(labels[i]+' '+locs[i],fontsize=15)
    if i == 2:
        axx.legend(loc=3,fontsize=14)

pl.tight_layout()
#pl.savefig(jashome+'temp_1906_obs/tmax_'+loc.replace(' ','')+'_'+year+'20CR_obs_ts.png')