# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:08:37 2019

@author: pmcraig
"""

import pylab as pl
import pandas as pd
import sklearn.metrics as skm
import matplotlib.gridspec as gridspec

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

years = pl.linspace(1900,1910,11).astype(int)

gridrain = pd.read_csv(ncasdir+'gridrain_18stations_1900-1910.csv',header=0)
dwrrain = pd.read_csv(ncasdir+'dwrrain_18stations_1900-1910.csv',header=0)

stations = gridrain.columns.values

gridrain = pl.asarray(gridrain)
dwrrain = pl.asarray(dwrrain)

#stat = 0
#pl.figure(figsize=(19,5))
##pl.plot(gridrain[:,stat],label='Met Office',lw=0,marker='x')
##pl.plot(dwrrain[:,stat]*25.4,label='Weather Rescue',lw=0,marker='.')
#pl.plot(gridrain[1:,stat]-dwrrain[:-1,stat])
#pl.xlim(0,dwrrain.shape[0]-1)
#pl.title(stations[stat])
##pl.legend(loc=2)
#pl.tight_layout()

gridyrs = [gridrain[:365],gridrain[365:730],gridrain[730:1095],
           gridrain[1095:1460],gridrain[1460:1826],gridrain[1826:2191],
            gridrain[2191:2556],gridrain[2556:2921],gridrain[2921:3287],
            gridrain[3287:3652],gridrain[3652:4016]]
dwryrs = [dwrrain[1:366],dwrrain[366:731],dwrrain[731:1096],dwrrain[1096:1461],
          dwrrain[1461:1827],dwrrain[1827:2192],dwrrain[2192:2557],
          dwrrain[2557:2922],dwrrain[2922:3288],dwrrain[3288:3653],
          dwrrain[3653:]]
#gridyrs = [gridrain[:365],gridrain[365:730],gridrain[730:1095],
#          gridrain[1095:1460],gridrain[1460:1826],gridrain[1826:2191],
#          gridrain[2191:2556],gridrain[2556:2921],gridrain[2921:3287],
#          gridrain[3287:3652],gridrain[3652:]]
#dwryrs = [dwrrain[:365],dwrrain[365:730],dwrrain[730:1095],
#          dwrrain[1095:1460],dwrrain[1460:1826],dwrrain[1826:2191],
#          dwrrain[2191:2556],dwrrain[2556:2921],dwrrain[2921:3287],
#          dwrrain[3287:3652],dwrrain[3652:]]

for stat in range(len(stations)):
    rmse = RMSE(dwrrain[1:,stat]*25.4,gridrain[:-1,stat])
    mbe = MBE(dwrrain[1:,stat]*25.4,gridrain[:-1,stat])
    print stations[stat], round(rmse,2), round(mbe,2)

#for stat in range(len(stations)):
#    fig = pl.figure(figsize=(10,10))
#    gs = gridspec.GridSpec(6, 4)
#    ig = [gs[0,0:2],gs[0,2:],gs[1,0:2],gs[1,2:],gs[2,0:2],gs[2,2:],
#          gs[3,0:2],gs[3,2:],gs[4,0:2],gs[4,2:],gs[5,1:3]]
#
#    
#    for yr in range(len(years)):
#        ax = pl.subplot(ig[yr])
#        #a = ax.plot(gridyrs[yr][:,stat],label='Met Office')
#        #b = ax.plot(dwryrs[yr][:,stat]*25.4,label='Weather Rescue',alpha=0.9)
#        ax.plot(dwryrs[yr][:,stat]*25.4-gridyrs[yr][:,stat])
#        pl.xlim(0,dwryrs[yr][:,stat].size)
#        pl.ylim(-50,50)
#        pl.yticks(pl.linspace(-50,50,5))
#        ax.grid(axis='y',ls='--')
#        pl.annotate(str(years[yr]),xy=(0.01,0.8),xycoords='axes fraction',fontsize=14)
#        #if yr in (0,2,4,6,8,10):
#        pl.ylabel('mm',fontsize=13,labelpad=-5)
#        if yr > 9:
#            pl.xlabel('days',fontsize=13)
#    
#    #fig.legend([a[0],b[0]],['Met Office','Weather Rescue'],loc=(0.05,0.1),fontsize=12)
#    pl.suptitle('Weather Rescue minus Met Office: '+stations[stat],fontsize=15)
#    pl.tight_layout()
#    pl.subplots_adjust(top=0.95)
#    
#    name = stations[stat].replace(' ','')
#    pl.savefig(jashome+'raincomp_plots_1km/11years/raindiffs_1900-1910_'+name+'.png')
