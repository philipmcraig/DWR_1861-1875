# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:33:19 2019

@author: pmcraig
"""

from __future__ import division
import pylab as pl
import glob
import pandas as pd
import matplotlib.gridspec as gridspec
import pcraig_funcs as pc

def MidasTemp(tempdir,county,station,year):
    """
    """
    filename = glob.glob(tempdir+county+'/'+station+'/qc-version-1/*'+year+'*.csv')
    data = pd.read_csv(filename[0],header=90)
    df = pd.DataFrame(data)
    time = df.ob_end_time
    maxtemps = df.max_air_temp
    
    temp_augsep = []; time_augsep = []
    for i in range(len(time)):
        if time[i][5:7] == months[0] or time[i][5:7] == months[1]:
            temp_augsep.append(maxtemps[i])
            time_augsep.append(time[i])
    temp_augsep = pl.asarray(temp_augsep)
    
    return temp_augsep

def DWRtemp(ncasdir,year,months,loc):
    """
    """
    dwraug = glob.glob(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+months[0]+'*')
    dwrsep = glob.glob(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+months[1]+'*')
    
    dwraug = dwraug[17:]; dwrsep = dwrsep[:16]
    dwrfiles = dwraug + dwrsep
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

def MakePlot(ax,midastemp,dwrtemp,loc,letter,i):
    """
    """
    ax.plot(midastemp[17:47],lw=3.5,label='MIDAS')
    ax.plot(dwrtemp,color='orange',lw=2,ls='-',label=' DWR')
    pl.xlim(0,len(midastemp[17:47])-1)
    pl.ylim(10,35); pl.grid(axis='y',ls='--')
    pl.ylabel('$^\circ$C',fontsize=16)
    pl.xticks([0,4,9,14,19,24,29])
    ax.set_xticklabels(['Aug 17','Aug 21','Aug 26','Aug 31','Sep 5','Sep 10','Sep 15'])
    ax.tick_params(axis='y',direction='in')
    ax.tick_params(axis='x',pad=7,direction='in',length=3,top=True)
    if i == 1:
        ax.legend(loc=1,fontsize=13)
    pl.title(letter+' '+loc,fontsize=15)
    
    return None

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
#CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
tempdir = '/badc/ukmo-midas-open/data/uk-daily-temperature-obs/dataset-version-201901/'
CR20dir = ncasdir + '20CR/'
year = '1906'
months = ['08','09']

counties = ['oxfordshire','merseyside','avon']
stations = ['00606_oxford','01128_bidston','01312_bath']
locs = ['Oxford','Liverpool','Bath']
letters = ['(a)','(b)','(c)']
#midastemps = []; dwrtemps = []

pl.figure(figsize=(12,8))
gs = gridspec.GridSpec(2, 4)
ig = [gs[0,:2],gs[0,2:],gs[1,1:3]]

for i in range(len(locs)):
    midastemps = MidasTemp(tempdir,counties[i],stations[i],year)
    dwrtemps = DWRtemp(ncasdir,year,months,locs[i])
    
    axx = pl.subplot(ig[i])
    MakePlot(axx,midastemps,dwrtemps,locs[i],letters[i],i)
    
pl.tight_layout()

################################################################################
#filename = glob.glob(tempdir+counties[0]+'/'+stations[0]+'/qc-version-1/*'+year+'*.csv')
#data = pd.read_csv(filename[0],header=90)
#df = pd.DataFrame(data)
#time = df.ob_end_time
#maxtemps = df.max_air_temp
#
#temp_augsep = []; time_augsep = []
#for i in range(len(time)):
#    if time[i][5:7] == months[0] or time[i][5:7] == months[1]:
#        temp_augsep.append(maxtemps[i])
#        time_augsep.append(time[i])
#temp_augsep = pl.asarray(temp_augsep)
################################################################################
#
#
################################################################################
#dwraug = glob.glob(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+months[0]+'*')
#dwrsep = glob.glob(ncasdir+'DWRcsv/'+year+'/dwr_'+year+'_'+months[1]+'*')
#
#dwraug = dwraug[17:]; dwrsep = dwrsep[:16]
#dwrfiles = dwraug + dwrsep
#dwrtemps = pl.zeros([len(dwrfiles)])
#
#for i in range(len(dwrfiles)):
#    logs = pd.read_csv(dwrfiles[i],header=None)
#    obs = pl.array(logs)
#    ind = pl.where(obs[:,0]==locs[0]+' '); ind = ind[0]
#    if obs[ind,-3] == ' ':
#         dwrtemps[i] = pl.float32('nan')
#    else:
#         dwrtemps[i] = pl.float32(obs[ind,-3])
#
#dwrtemps = (dwrtemps-32)*(5./9.)
################################################################################
#
#
#fig, ax = pl.subplots()
#ax.plot(temp_augsep[17:47],lw=2,label='MIDAS')
#ax.plot(dwrtemps,color='orange',lw=2,ls='-',label=' DWR observations')
#pl.xlim(0,len(temp_augsep[17:47])-1)
#pl.ylim(10,35); pl.grid(axis='y',ls='--')
#pl.ylabel('$^\circ$C',fontsize=16)
#pl.xticks([0,4,9,14,19,24,29])
#ax.set_xticklabels(['Aug 17','Aug 21','Aug 26','Aug 31','Sep 5','Sep 10','Sep 15'])
#ax.tick_params(axis='y',direction='in')
#ax.tick_params(axis='x',pad=7,direction='in',length=3,top=True)
#ax.legend()
#pl.title(locs[2],fontsize=15)
#
#pl.tight_layout()
##pl.savefig(jashome+'temp_1906_obs/tmax_'+locs[1].replace(' ','')+'_'+year+'MIDAS_DWR_ts.png')