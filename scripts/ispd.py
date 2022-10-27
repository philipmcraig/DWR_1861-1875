# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:34:10 2019

@author: pmcraig
"""

import pylab as pl
import pandas as pd
import glob

pl.close('all')

ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
ispddir = ncasdir + 'ISPD/'
dwrdir = ncasdir + 'DWRcsv/'

year = '1910'
dwrfiles = glob.glob(dwrdir+year+'/*')
dwrfiles = pl.sort(dwrfiles)
evepres = pl.zeros([len(dwrfiles)])
morpres = pl.zeros([len(dwrfiles)])

ind = 67
for i in range(len(dwrfiles)):
    df = pd.read_csv(dwrfiles[i],header=None)
    logs = pl.array(df)
    if logs[ind,1] == '  ' or logs[ind,1] == ' ':
        evepres[i] = pl.float64('nan')
    else:
        evepres[i] = pl.float64(logs[ind,1])*33.8639
    if logs[ind,3] == '  ' or logs[ind,3] == ' ':
        morpres[i] = pl.float64('nan')
    else:
        morpres[i] = pl.float64(logs[ind,3])*33.8639

loc = 'Madiera'
with open(ispddir+year+'/'+loc+year+'.txt','r') as f:
    data = f.readlines()

pres1 = pl.zeros([len(data)])
pres2 = pl.zeros([len(data)])
times = pl.zeros([len(data)],dtype='S12')

for i in range(len(data)):
    times[i] = data[i][:12]
    pres1[i] = pl.float64(data[i][55:62])
    pres2[i] = pl.float64(data[i][64:71])

A = 3; H = pl.array([10,16,22])
newarray = pl.zeros([12,31,A])
for i in range(len(times)):
    mon = int(times[i][4:6]); ind0 = mon - 1
    day = int(times[i][6:8]); ind1 = day - 1
    hour = int(times[i][8:10])
    a = pl.where(H==hour); ind2 = a[0][0]
    newarray[ind0,ind1,ind2] = pres1[i]

P = pl.zeros([365,A])
#jan = newarray[0].flatten(); feb = newarray[1].flatten()
#mar = newarray[2].flatten(); apr = newarray[3].flatten()
#may = newarray[4].flatten(); jun = newarray[5].flatten()
#jul = newarray[6].flatten(); aug = newarray[7].flatten()
#sep = newarray[8].flatten(); ocr = newarray[9].flatten()
#nov = newarray[10].flatten(); dec = newarray[11].flatten()
P[:31] = newarray[0,:]; P[31:59] = newarray[1,:28]
P[59:90] = newarray[2,:]; P[90:120] = newarray[3,:30]
P[120:151] = newarray[4,:]; P[151:181] = newarray[5,:30]
P[181:212] = newarray[6,:]; P[212:243] = newarray[7,:]
P[243:273] = newarray[8,:30]; P[273:304] = newarray[9,:]
P[304:334] = newarray[10,:30]; P[334:] = newarray[11,:]
P[P==0] = pl.float32('nan')

#pres1 = pl.reshape(pres1,(pres1.shape[0]/A,A))
#pres2 = pl.reshape(pres2,(pres2.shape[0]/A,A))

dwrdata = morpres[:]
#dwrdata = evepres[1:]

fig, ax = pl.subplots(1,1,figsize=(14,9))
pl.grid(axis='y',ls='-',color='lightgrey')
pl.plot(P[:,0],label='ISPD 1000')
pl.plot(dwrdata,label='DWR 0800')
pl.legend()
pl.xlim(0,364)
pl.ylabel('hPa',fontsize=15)
labels = ['1','','','','','','7',
          '','','','','','','14',
          '','','','','','','21',
          '','','','','','','28']
#pl.xticks(pl.linspace(0,27,28),labels)
ends = [30,58,89,119,150,180,211,242,272,303,333]
midpts = pl.array([15,44,73,104,134,165,195,226,257,287,318,349])
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for i in range(len(ends)):
    pl.axvline(x=ends[i],ls='--',color='k',lw=0.5,zorder=0)
ax.tick_params(axis='x',length=0)
pl.xticks(midpts,months,fontsize=14)
pl.title(loc+' '+year)
#pl.savefig(ispddir+'/'+year+'/'+'ispd_dwr_mslp_'+loc+'_'+year+'_0800.png')