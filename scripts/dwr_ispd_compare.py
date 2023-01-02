# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:10:19 2022

@author: qx911590
"""

import pylab as pl
import pandas as pd
import glob

def GW_data(statname,sefdir,ispddir):
    """
    """
    dwrsef = glob.glob(sefdir+'DWR_UKMO_DWRUK_'+statname+'*'+'_mslp.tsv')
    dwrdata = pl.genfromtxt(dwrsef[0],skiprows=13)
    dwrpres = dwrdata[:,-2]
    dwryr = dwrdata[:,0].astype(int)
    dwrmn = dwrdata[:,1].astype(int)
    dwrdy = dwrdata[:,2].astype(int)
    
    ispdsef = glob.glob(ispddir+'ISPDv4_EU/ISPDv4_GALWAY_p_*.tsv')
    ispddata = pl.genfromtxt(ispdsef[0],delimiter='	',skiprows=13,dtype='object')
    ispdpres = ispddata[:,-2].astype(float)
    ispdyr = ispddata[:,0].astype(int)
    ispdmn = ispddata[:,1].astype(int)
    ispddy = ispddata[:,2].astype(int)
    ispdhr = ispddata[:,3].astype(int)
    ispdmt = ispddata[:,-1]
    
    start = pl.where((ispdyr==dwryr[0]) & (ispdmn==dwrmn[0]) & (ispddy==dwrdy[0]))[0][0]
    end = pl.where((ispdyr==dwryr[-1]) & (ispdmn==dwrmn[-1]) & (ispddy==dwrdy[-1]))[0][0]
    
    when_qc = pl.where(ispdmt!='')[0]
    qc_dates = pl.zeros([when_qc.size],dtype='S10')
    for i in range(when_qc.size):
        qc_dates[i] = str(ispdyr[when_qc[i]])+\
                                    '/'+"{0:0=2d}".format(ispdmn[when_qc[i]])+\
                                    '/'+"{0:0=2d}".format(ispddy[when_qc[i]])
    
    dwrpres_ext = pl.zeros_like(ispdpres[start:end+1])
    
    dates_dwr = pl.zeros_like(dwryr,dtype='S10')
    for i in range(len(dates_dwr)):
        dates_dwr[i] = str(dwryr[i])+'/'+"{0:0=2d}".format(dwrmn[i])+\
                                                '/'+"{0:0=2d}".format(dwrdy[i])

    cols = []
    inds = []
    plotdates = []
    zo = pl.zeros([ispdyr[start:end+1].size])
    
    for i in range(ispdyr[start:end+1].size):
        date = str(ispdyr[start:end+1][i])+'/'+\
                          "{0:0=2d}".format(ispdmn[start:end+1][i])+'/'+\
                                      "{0:0=2d}".format(ispddy[start:end+1][i])
        plotdates.append(date)
        
        if date not in dates_dwr:
            dwrpres_ext[i] = pl.float16('nan')
        else:
            a = pl.where(dates_dwr==date)[0][0]
            dwrpres_ext[i] = dwrpres[a]
        
        if date in qc_dates:
            cols.append('r')
            inds.append(i)
            zo[i] = 5
        else:
            cols.append('b')
            zo[i] = 1
    
    return dwrpres_ext, ispdpres[start:end+1], cols, plotdates, zo.astype(int)

def PM_data(statname,sefdir,ispddir):
    """
    """
    dwrsef = glob.glob(sefdir+'DWR_UKMO_DWRUK_'+statname+'*'+'_mslp.tsv')
    dwrdata = pl.genfromtxt(dwrsef[0],skiprows=13)
    dwrpres = dwrdata[:,-2]
    dwryr = dwrdata[:,0].astype(int)
    dwrmn = dwrdata[:,1].astype(int)
    dwrdy = dwrdata[:,2].astype(int)
    
    ispdsef = glob.glob(ispddir+'ISPDv4_EU/ISPDv4_PLYMOUTH-UK_mslp_*.tsv')
    ispddata = pl.genfromtxt(ispdsef[0],delimiter='	',skiprows=13,dtype='object')
    ispdpres = ispddata[:,-2].astype(float)
    ispdyr = ispddata[:,0].astype(int)
    ispdmn = ispddata[:,1].astype(int)
    ispddy = ispddata[:,2].astype(int)
    ispdhr = ispddata[:,3].astype(int)
    ispdmt = ispddata[:,-1]
    
    start = pl.where((ispdyr==dwryr[0]) & (ispdmn==dwrmn[0]) & (ispddy==dwrdy[0]))[0][0]
    end = pl.where((ispdyr==dwryr[-1]) & (ispdmn==dwrmn[-1]) & (ispddy==dwrdy[-1]))[0][0]
    
    when_qc = pl.where(ispdmt!='')[0]
    qc_dates = pl.zeros([when_qc.size],dtype='S10')
    for i in range(when_qc.size):
        qc_dates[i] = str(ispdyr[when_qc[i]])+\
                                    '/'+"{0:0=2d}".format(ispdmn[when_qc[i]])+\
                                    '/'+"{0:0=2d}".format(ispddy[when_qc[i]])
    
    dwrpres_ext = pl.zeros_like(ispdpres[start:end+1])
    
    dates_dwr = pl.zeros_like(dwryr,dtype='S10')
    for i in range(len(dates_dwr)):
        dates_dwr[i] = str(dwryr[i])+'/'+"{0:0=2d}".format(dwrmn[i])+\
                                                '/'+"{0:0=2d}".format(dwrdy[i])

    cols = []
    inds = []
    plotdates = []
    
    for i in range(ispdyr[start:end+1].size):
        date = str(ispdyr[start:end+1][i])+'/'+\
                          "{0:0=2d}".format(ispdmn[start:end+1][i])+'/'+\
                                      "{0:0=2d}".format(ispddy[start:end+1][i])
        plotdates.append(date)
        
        if date not in dates_dwr:
            dwrpres_ext[i] = pl.float16('nan')
        else:
            a = pl.where(dates_dwr==date)[0][0]
            dwrpres_ext[i] = dwrpres[a]
        
        if date in qc_dates:
            cols.append('r')
            inds.append(i)
        else:
            cols.append('b')
    
    return dwrpres_ext, ispdpres[start:end+1], cols, plotdates

def HC_data(statname,sefdir,ispddir):
    """
    """
    dwrsef = glob.glob(sefdir+'DWR_UKMO_DWRUK_'+statname+'*'+'_mslp.tsv')
    dwrdata = pl.genfromtxt(dwrsef[0],skiprows=13)
    dwrpres = dwrdata[:,-2]
    dwryr = dwrdata[:,0].astype(int)
    dwrmn = dwrdata[:,1].astype(int)
    dwrdy = dwrdata[:,2].astype(int)
    
    ispdsef = glob.glob(ispddir+'ISPDv4_can/ISPDv4_ST-JOHNS_p_18660101_18700831.tsv')
    ispddata = pl.genfromtxt(ispdsef[0],delimiter='	',skiprows=13,dtype='object')
    ispdpres = ispddata[:,-2].astype(float)
    ispdyr = ispddata[:,0].astype(int)
    ispdmn = ispddata[:,1].astype(int)
    ispddy = ispddata[:,2].astype(int)
    ispdhr = ispddata[:,3].astype(int)
    ispdmt = ispddata[:,-1]
    
    when12 = pl.where(ispdhr==12)[0]
    when18 = pl.where(ispdhr==18)[0]
    when_qc = pl.where((ispdmt!='') & (ispdhr==12))[0]
    
    qc_dates = pl.zeros([when_qc.size],dtype='S10')
    for i in range(when_qc.size):
        qc_dates[i] = str(ispdyr[when_qc[i]])+\
                                    '/'+"{0:0=2d}".format(ispdmn[when_qc[i]])+\
                                    '/'+"{0:0=2d}".format(ispddy[when_qc[i]])

    ispdpres_12 = ispdpres[when12]
    ispdyr_12 = ispdyr[when12]; ispdmn_12 = ispdmn[when12]; ispddy_12 = ispddy[when12]
    
#    ispdpres_18 = ispdpres[when18]
#    ispdyr_18 = ispdyr[when18]; ispdmn_18 = ispdmn[when18]; ispddy_18 = ispddy[when18]
    
    start12_ispd =  pl.where((ispdyr_12==1868) & (ispdmn_12==1) & (ispddy_12==13))[0][0]
    #start18_ispd =  pl.where((ispdyr[when18]==1868) & (ispdmn[when18]==1) & (ispddy[when18]==13))[0][0]
    end_dwr = pl.where((dwryr==1870) & (dwrmn==8) & (dwrdy==31))[0][0]
    
    dwrpres_ext = pl.zeros_like(ispdpres_12[start12_ispd:])
    
    dates_dwr = pl.zeros_like(dwryr[:end_dwr+1],dtype='S10')
    for i in range(len(dates_dwr[:end_dwr+1])):
        dates_dwr[i] =  str(dwryr[:end_dwr+1][i])+'/'+\
                                    "{0:0=2d}".format(dwrmn[:end_dwr+1][i])+\
                                    '/'+"{0:0=2d}".format(dwrdy[:end_dwr+1][i])
    
    cols = []
    inds = []
    plotdates = []
    
    for i in range(len(ispdyr_12[start12_ispd:])):
        date = str(ispdyr_12[start12_ispd:][i])+'/'+\
                        "{0:0=2d}".format(ispdmn_12[start12_ispd:][i])+'/'+\
                                "{0:0=2d}".format(ispddy_12[start12_ispd:][i])
        plotdates.append(date)
    #    a = pl.where(dates==date)
    #    if a[0].size!=1:
    #        print date
        if date not in dates_dwr:
            dwrpres_ext[i] = pl.float16('nan')
            #print date
        else:
            a = pl.where(dates_dwr==date)[0][0]
            dwrpres_ext[i] = dwrpres[a]
        
        if date in qc_dates:
            cols.append('r')
            inds.append(i)
        else:
            cols.append('b')
    
    return dwrpres_ext, ispdpres_12[start12_ispd:], cols, plotdates

pl.close('all')

sefdir = 'C:\Users\phili\GitHub\DWR_1861-1875\SEF_1861-1875/'
ispddir = 'C:\Users\phili/GitHub/weather-rescue/'
emulate = 'C:\Users\phili/GitHub/weather-rescue/emulate/'
#iredir = 'C:\Users\qx911590\Documents/ILMMT/'

stations = ['GALWAY','PLYMOUTH','HEARTSCONTENT']

#dwrsef = glob.glob(sefdir+'DWR_UKMO_DWRUK_'+station+'*'+'_mslp.tsv')
#dwrdata = pl.genfromtxt(dwrsef[0],skiprows=13)
#dwrpres = dwrdata[:,-2]
#dwryr = dwrdata[:,0].astype(int)
#dwrmn = dwrdata[:,1].astype(int)
#dwrdy = dwrdata[:,2].astype(int)
#
#ispdsef = glob.glob(ispddir+'ISPDv4_ST-JOHNS_p_18660101_18700831.tsv')
#ispddata = pl.genfromtxt(ispdsef[0],delimiter='	',skiprows=13,dtype='object')
#ispdpres = ispddata[:,-2].astype(float)
#ispdyr = ispddata[:,0].astype(int)
#ispdmn = ispddata[:,1].astype(int)
#ispddy = ispddata[:,2].astype(int)
#ispdhr = ispddata[:,3].astype(int)
#ispdmt = ispddata[:,-1]



#irefiles = iredir+'Long-term-series/NUIGalway/'
#iredata = pl.genfromtxt(irefiles+'NUIGalway_1851-1965.csv',skiprows=1)
#iremax = iredata[:,-2]
#iremin = iredata[:,-1]

#start = pl.where((ispdyr==dwryr[0]) & (ispdmn==dwrmn[0]) & (ispddy==dwrdy[0]))[0][0]
#end = pl.where((dwryr==ispdyr[-1]) & (dwrmn==ispdmn[-1]) & (dwrdy==ispddy[-1]))[0][0]



#dates_ispd = pl.zeros_like(ispdyr[start:end+1],dtype='S10')
#for i in range(len(dates_ispd)):
#    dates_ispd[i] = str(ispdyr[start:end+1][i])+'/'+\
#                                    "{0:0=2d}".format(ispdmn[start:end+1][i])+\
#                                    '/'+"{0:0=2d}".format(ispddy[start:end+1][i])

dwrpres_gw, ispdpres_gw, cols_gw, plotdates_gw, zo_gw = GW_data(stations[0],sefdir,ispddir)

dwrpres_pm, ispdpres_pm, cols_pm, plotdates_pm = PM_data(stations[1],sefdir,ispddir)

dwrpres_hc, ispdpres_hc, cols_hc, plotdates_hc = HC_data(stations[-1],sefdir,ispddir)


fig, ax = pl.subplots(3,1,figsize=(20,12))

ax1 = pl.subplot(311)

ax1.scatter(pl.linspace(0,1346,1346),dwrpres_gw-ispdpres_gw,s=30,marker='o',
            c=pl.array(cols_gw),edgecolors=cols_gw,facecolor=cols_gw)#,zorder=zo_gw)
pl.xlim(0,1345); pl.ylim(-20,30)
pl.yticks(size=13); pl.ylabel('hPa',fontsize=16,labelpad=-10)
pl.grid(axis='y')
pl.title('Galway DWR mslp minus ISPD p')
pl.xticks(pl.linspace(1,1345,1346)[::100],plotdates_gw[::100])
fig.text(0.0075,0.955,'(a)',size=15)

qcr = pl.where(pl.asarray(cols_gw)=='r')[0]
dot = ax1.scatter(qcr,dwrpres_gw[qcr]-ispdpres_gw[qcr],s=40,marker='o',
                  c='r',edgecolors='r',facecolor='r')
###############################################################################

ax2 = pl.subplot(312)

sct = ax2.scatter(pl.linspace(0,3117,3118),dwrpres_pm-ispdpres_pm,s=30,marker='o',
            c=pl.array(cols_pm),edgecolors=cols_pm,facecolor=cols_pm)
pl.xlim(0,3117); pl.ylim(-5,5)
pl.yticks(size=13); pl.ylabel('hPa',fontsize=16,labelpad=-10)
pl.grid(axis='y')
pl.title('Plymouth DWR mslp minus ISPD mslp')
pl.xticks(pl.linspace(1,3117,3118)[::200],plotdates_pm[::200])
fig.text(0.0075,0.62,'(b)',size=15)

qcr = pl.where(pl.asarray(cols_pm)=='r')[0]
dot = ax2.scatter(qcr,dwrpres_pm[qcr]-ispdpres_pm[qcr],s=40,marker='o',
                  c='r',edgecolors='r',facecolor='r')

ax2.legend((sct,dot),('qc accept','qc reject'),loc=0)
#ax2.legend(*sct.legend_elements(num=2),loc=0,handles=['normal','reject'])

###############################################################################

ax3 = pl.subplot(313)#pl.figure(figsize=(23.5,8))
#pl.plot(dwrpres_ext,lw=0,label='DWR',marker='.')
#pl.plot(ispdpres_12[start12_ispd:],lw=0,label='ISPD',marker='x')
#pl.plot(dwrpres_ext-ispdpres_12[start12_ispd:],marker='.',ms=10,lw=0)
ax3.scatter(pl.linspace(1,960,960),dwrpres_hc-ispdpres_hc,s=30,zorder=1,
           marker='o',c=pl.array(cols_hc),edgecolors=cols_hc,facecolor=cols_hc)
ax3.axvline(x=294,ls='--',color='k',label='DWR time change')
#pl.xlim(0,1345)
ax3.legend(loc=1)

pl.xlim(0,960); pl.ylim(-40,60)
pl.yticks(size=13); pl.ylabel('hPa',fontsize=16,labelpad=-10)
pl.grid(axis='y')
pl.title('Hearts Content DWR mslp minus St Johns ISPD 1200 p')
pl.text(150,52,'DWR 0930',fontsize=16)
pl.text(470,52,'DWR 1230',fontsize=16)
pl.xticks(pl.linspace(1,960,960)[::100],plotdates_hc[::100])
fig.text(0.0075,0.285,'(c)',size=15)

qcr = pl.where(pl.asarray(cols_hc)=='r')[0]
dot = ax3.scatter(qcr,dwrpres_hc[qcr]-ispdpres_hc[qcr],s=40,marker='o',
                  c='r',edgecolors='r',facecolor='r',zorder=30)

pl.tight_layout()
pl.subplots_adjust(bottom=0.02,top=0.98,hspace=0.15)

#pl.savefig('C:\Users\qx911590\Documents/figures/dwr_ispd_pres_compare_panels.png',dpi=400)
#pl.savefig('C:\Users\qx911590\Documents/figures/dwr_ispd_pres_compare_panels.pdf',dpi=400)