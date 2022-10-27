# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:20:44 2022

@author: qx911590
"""

from __future__ import division
import pylab as pl
import glob
import pandas as pd
import xarray as xr
import pcraig_funcs as pc

def MDS_func(filepath):
    """
    """
    df = pd.read_csv(filepath,skiprows=90)
    df = df.head(df.shape[0] -1)
    mds_arr = pl.asarray(df)
    mds_dates = pl.asarray([x[:10] for x in mds_arr[:,0]])
    Tx_mds = mds_arr[:,8]
    Tn_mds = mds_arr[:,9]
    
    return Tx_mds, Tn_mds, mds_dates
    

def SEF_maxmin(seffiles,ind,mds_dates):
    """
    """
    df = pd.read_csv(seffiles[ind],skiprows=13,delimiter='	')
    sef_arr = pl.asarray(df)
    
    dwr_dates = pl.asarray([str(x[0])+'-'+"{:02d}".format(x[1])+'-'\
                                    +"{:02d}".format(x[2]) for x in sef_arr[:]])

    temp = sef_arr[:,-2]
    temp_ext = pl.zeros([mds_dates.size]); temp_ext[:] = float('nan')
    
    for i in range(mds_dates.size):
        if mds_dates[i] in dwr_dates:
            F = pl.where(dwr_dates==mds_dates[i])[0][0]
            temp_ext[i] = temp[F]
    
    return temp_ext

def IRE_func(iredir):
    """
    """
    df = pd.read_csv(iredir+'NUIGalway_1851-1965.csv',skiprows=0)
    ire_arr = pl.asarray(df)
    yr_inds = pl.where(ire_arr[:,0]==1863)[0]
    ire_dates = pl.asarray([str(int(x[0]))+'-'+"{:02d}".format(int(x[1]))+'-'+\
            "{:02d}".format(int(x[2])) for x in ire_arr[yr_inds[0]:yr_inds[-1]+1]])
    Tx_ire = ire_arr[yr_inds[0]:yr_inds[-1]+1,-2]
    Tn_ire = ire_arr[yr_inds[0]:yr_inds[-1]+1,-1]
    
    return Tx_ire, Tn_ire, ire_dates

def SEF_tdry(seffiles,ire_dates):
    """
    """
    df = pd.read_csv(seffiles[0],skiprows=13,delimiter='	')
    sef_arr = pl.asarray(df)
    dwr_dates = pl.asarray([str(x[0])+'-'+"{:02d}".format(x[1])+'-'\
                                 +"{:02d}".format(x[2]) for x in sef_arr[:]])
    temp = sef_arr[:,-2]
    temp_ext = pl.zeros([ire_dates.size]); temp_ext[:] = float('nan')
    for i in range(mds_dates.size):
        if ire_dates[i] in dwr_dates:
            F = pl.where(dwr_dates==ire_dates[i])[0][0]
            temp_ext[i] = temp[F]
    
    return temp_ext

pl.close('all')

#homedir = '/home/users/qx911590/'
jashome = '/home/users/pmcraig/'
midasdir = '/badc/ukmo-midas-open/data/uk-daily-temperature-obs/dataset-version-202107/'#homedir + 'weatherrescue/midasfiles/'
sefdir = jashome + 'seffiles/'
iredir = jashome + 'ILMMT/'
CR20dir = '/gws/nopw/j04/glosat/development/data/raw/20CRv3/451/subdaily/'

year = '1873'
counties = ['merseyside','oxfordshire']
mds_locs = ['bidston','oxford']
sef_locs = ['LIVERPOOL','OXFORD','GALWAY']

cr20files_a = glob.glob(CR20dir+'1873/TMP2m*')
cr20files_b = glob.glob(CR20dir+'1863/TMP2m*')
enslen = len(cr20files_a)
temp_gw = pl.zeros([enslen,2912])
temp_lp = temp_gw.copy()
temp_ox = temp_gw.copy()

for nc in range(enslen):
    ncfile = xr.open_dataset(cr20files_a[nc])
    if nc == 0:
        lat = xr.DataArray(ncfile.lat[:64]).data
        lon = xr.DataArray(ncfile.lon[490:]).data
        time = xr.DataArray(ncfile.time).data
        
        lp_latind = pc.NearestIndex(lat,53.400735)
        lp_lonind = pc.NearestIndex(lon,-3.074226)
        
        ox_latind = pc.NearestIndex(lat,51.7612)
        ox_lonind = pc.NearestIndex(lon,360-1.26399)
        
        gw_latind = pc.NearestIndex(lat,53.27866)
        gw_lonind = pc.NearestIndex(lon,360-9.06126)

    temp_lp[nc,:] = pl.squeeze(xr.DataArray(ncfile.TMP2m[3:-5,:,lp_latind,lp_lonind]).data)
    temp_ox[nc,:] = pl.squeeze(xr.DataArray(ncfile.TMP2m[3:-5,:,ox_latind,ox_lonind]).data)
    
    ncfile.close()
    
    ncfile = xr.open_dataset(cr20files_b[nc])
    temp_gw[nc,:] = pl.squeeze(xr.DataArray(ncfile.TMP2m[3:-5,:,gw_latind,gw_lonind]).data)
    ncfile.close()

fig, ax = pl.subplots(3,1,figsize=(16,8))

for i in range(len(mds_locs)):

    midasfiles = glob.glob(midasdir+counties[i]+'/*'+mds_locs[i]+'/qc-version-1/'+
                            'midas-open_uk-daily-temperature-obs_dv-202107_*'+\
                                                mds_locs[i]+'*'+year+'.csv')
    
    seffiles = glob.glob(sefdir+'*'+sef_locs[i]+'*')

    Tx_mds, Tn_mds, mds_dates = MDS_func(midasfiles[0])
    if i == 0:
        Tx_dwr = SEF_maxmin(seffiles,1,mds_dates)
        Tn_dwr = SEF_maxmin(seffiles,0,mds_dates)
        TD = pl.reshape(temp_lp,newshape=(80,364,8))
    elif i == 1:
        Tx_dwr = SEF_maxmin(seffiles,0,mds_dates)
        Tn_dwr = SEF_maxmin(seffiles,1,mds_dates)
        TD = pl.reshape(temp_ox,newshape=(80,364,8))
    
    X = TD.max(axis=2) - 273.15
    N = TD.min(axis=2) - 273.15
    
    mx_20cr = pl.full(shape=(365,),fill_value=float('nan'))
    mx_20cr[1:] = pl.mean(X,axis=0)
    mn_20cr = pl.full(shape=(365,),fill_value=float('nan'))
    mn_20cr[1:] = pl.mean(N,axis=0)


    axx = pl.subplot(3,1,i+1)
    axx.plot(Tx_dwr-Tx_mds,marker='o',lw=0.5,label='Tmax',ms=6)
    axx.plot(Tx_dwr-mx_20cr,lw=0.5,color='k',zorder=5,label='20CRv3 Tmax')
    axx.plot(Tn_dwr[1:]-Tn_mds[:-1],marker='o',lw=0.5,label='Tmin',ms=4)
    axx.plot(Tn_dwr-mn_20cr,lw=0.5,color='k',ls='--',zorder=5,label='20CRv3 Tmin')
    axx.grid(axis='y',ls=':',color='grey')
    pl.xlim(0,364)
    pl.xticks([0,31,59,90,120,151,181,212,243,273,304,334,364])
    axx.set_xticklabels(mds_dates[[0,31,59,90,120,151,181,212,243,273,304,334,364]])
    pl.ylabel('$^\circ$C',fontsize=12,labelpad=-5)
    pl.title(sef_locs[i]+' '+'DWR minus MIDAS & 20CR Tmax, Tmin '+year,size=10)
    
    if i == 0:
        axx.legend(loc=2,ncol=2,columnspacing=0.5,handletextpad=0.5)

fig.text(0.005,0.95,'(a)',size=12)
fig.text(0.005,0.62,'(b)',size=12)

Tx_ire, Tn_ire, ire_dates = IRE_func(iredir)

seffiles = glob.glob(sefdir+'*'+sef_locs[2]+'*')
tdry_dwr = SEF_tdry(seffiles,ire_dates)




TD = pl.reshape(temp_gw,newshape=(80,364,8))
X = TD.max(axis=2)
N = TD.min(axis=2)


gw_mx = pl.mean(X,axis=0)
gw_mn = pl.mean(N,axis=0)

dflt_cl = pl.rcParams['axes.prop_cycle'].by_key()['color']

ax3 = pl.subplot(313)
ax3.plot(Tx_ire,lw=2,label='Tmax')
ax3.plot(Tn_ire,lw=2,label='Tmin')
ax3.plot(tdry_dwr,lw=2,label='Tdry')
ax3.plot(gw_mx-273.15,label='20CRv3 Tmax',color=dflt_cl[0],alpha=0.5)
ax3.plot(gw_mn-273.15,label='20CRv3 Tmin',color=dflt_cl[1],alpha=0.5)
ax3.grid(axis='y',ls=':',color='grey')
pl.xlim(0,364)
pl.xticks([0,31,59,90,120,151,181,212,243,273,304,334,364])
ax3.set_xticklabels(ire_dates[[0,31,59,90,120,151,181,212,243,273,304,334,364]])
pl.ylabel('$^\circ$C',fontsize=12)
pl.title(sef_locs[2]+' DWR Tdry & Mateus et al. (2020), 20CRv3 Tmax, Tmin 1863',size=10)

ax3.legend(loc=2,ncol=2,columnspacing=0.5,handletextpad=0.5)

fig.text(0.005,0.29,'(c)',size=12)

pl.tight_layout()
pl.subplots_adjust(top=0.97,bottom=0.04,wspace=0.16,hspace=0.23,right=0.972)

#pl.savefig(jashome+'midas_dwr_20cr_temps_comp_3stations.png',dpi=400)
#pl.savefig(jashome+'midas_dwr_20cr_temps_comp_3stations.pdf',dpi=400)