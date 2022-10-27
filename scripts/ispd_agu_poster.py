# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:15:56 2019

@author: pmcraig
"""

import pylab as pl
import pandas as pd
import glob

def RMSE(obs,mod):
    """Calculate root mean squared error and take nan values into account.
    
    Args:
        obs (array): observed data
        mod (array): predicted/modelled data
    
    Returns:
        rmse (float): root mean square error of data
    """
    z = pl.where(pl.isnan(obs)==True)
    w = pl.where(pl.isnan(mod)==True)
    v = z[0].tolist() + list(set(w[0].tolist()) - set(z[0].tolist()))
    v = pl.asarray(v)
    obs = pl.delete(obs,v)
    mod = pl.delete(mod,v)
    
    #rmse = pl.sqrt(skm.mean_squared_error(obs,mod))
    rmse = pl.sqrt(((mod-obs)**2).mean())
    
    return rmse


pl.close('all')
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
ispddir = ncasdir + 'ISPD/'
dwrdir = ncasdir + 'DWRcsv/'

years = [1903,1903]
loc = ['Aberdeen','Stornoway']

fig, ax = pl.subplots(2,1,figsize=(14,9))
ends = [30,58,89,119,150,180,211,242,272,303,333]
midpts = pl.array([15,44,73,104,134,165,195,226,257,287,318,349])
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
letters = ['(a) ','(b) ']

for i in range(len(years)):
    dwrfiles = glob.glob(dwrdir+str(years[i])+'/*')
    dwrfiles = pl.sort(dwrfiles)
    evepres = pl.zeros([len(dwrfiles)])
    morpres = pl.zeros([len(dwrfiles)])
    df = pd.read_csv(dwrfiles[i],header=None)
    logs = pl.array(df)
    ind = pl.where(logs[:,0]==loc[i]+' '); ind = ind[0][0]
    
    for j in range(len(dwrfiles)):
        df = pd.read_csv(dwrfiles[j],header=None)
        logs = pl.array(df)
        if logs[ind,1] == '  ' or logs[ind,1] == ' ':
            evepres[j] = pl.float64('nan')
        else:
            evepres[j] = pl.float64(logs[ind,1])*33.8639
        if logs[ind,3] == '  ' or logs[ind,3] == ' ':
            morpres[j] = pl.float64('nan')
        else:
            morpres[j] = pl.float64(logs[ind,3])*33.8639
    
    with open(ispddir+str(years[i])+'/'+loc[i]+str(years[i])+'.txt','r') as f:
        data = f.readlines()

    pres1 = pl.zeros([len(data)])
    pres2 = pl.zeros([len(data)])
    times = pl.zeros([len(data)],dtype='S12')
#    
    for k in range(len(data)):
        times[k] = data[k][:12]
        pres1[k] = pl.float64(data[k][55:62])
        pres2[k] = pl.float64(data[k][64:71])
    if i == 0:
        A = 4; H = pl.array([0,6,12,18])
    elif i == 1:
        A = 2; H = pl.array([9,21])
    newarray = pl.zeros([12,31,A])

    for t in range(len(times)):
        mon = int(times[t][4:6]); ind0 = mon - 1
        day = int(times[t][6:8]); ind1 = day - 1
        hour = int(times[t][8:10])
        a = pl.where(H==hour); ind2 = a[0][0]
        newarray[ind0,ind1,ind2] = pres1[t]
    
    P = pl.zeros([365,A])
    P[:31] = newarray[0,:]; P[31:59] = newarray[1,:28]
    P[59:90] = newarray[2,:]; P[90:120] = newarray[3,:30]
    P[120:151] = newarray[4,:]; P[151:181] = newarray[5,:30]
    P[181:212] = newarray[6,:]; P[212:243] = newarray[7,:]
    P[243:273] = newarray[8,:30]; P[273:304] = newarray[9,:]
    P[304:334] = newarray[10,:30]; P[334:] = newarray[11,:]
    P[P==0] = pl.float32('nan')
    
    if i == 1:
        dwrdata = morpres[:]
    elif i == 0:
        dwrdata = evepres[1:]
    

    axx = pl.subplot(2,1,i+1)
    pl.grid(axis='y',ls='-',color='lightgrey')
    if i == 0:
        axx.plot(P[:-1,-1],label='ISPD 1800')
        axx.plot(dwrdata,label='DWR 1800')
    elif i == 1:
        axx.plot(P[:,0],label='ISPD 0900')
    #if i == 0:
        axx.plot(dwrdata,label='DWR 0800')
    #elif i == 1:
    #    axx.plot(dwrdata,label='DWR 0700')
    pl.legend(loc=8,fontsize=14)
    pl.xlim(0,364)
    pl.ylabel('hPa',fontsize=15)
    for e in range(len(ends)):
        pl.axvline(x=ends[e],ls='--',color='k',lw=0.5,zorder=0)
    axx.tick_params(axis='x',length=0)
    pl.xticks(midpts,months,fontsize=14)
    
    if i == 0:
        rmse = RMSE(dwrdata,P[:-1,-1])
    elif i == 1:
        rmse = RMSE(dwrdata,P[:,0])
    pl.title(letters[i]+loc[i]+' '+str(years[i])+', RMSE = '+str(round(rmse,2)))

pl.tight_layout()
#pl.savefig(ispddir+loc+'_'+str(years[0])+'_'+str(years[1])+'_panels.png')
pl.savefig(ispddir+'abdn_snwy_panels_1903_poster.png')