# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:03:18 2022

@author: Philip
"""

import pylab as pl
import pandas as pd
import glob
import os

homedir = 'C:/Users/phili/Github/DWR_1861-1875/'
sefdir = 'C:/Users/phili/Github/DWR_1861-1875/SEF_1861-1875/'

mslp_files = glob.glob(sefdir+'*mslp.tsv')
X = pl.zeros([len(mslp_files)])
mslp_count = 0

tdry_files = glob.glob(sefdir+'*ta.tsv')
tdry_count = 0

twet_files = glob.glob(sefdir+'*tb.tsv')
twet_count = 0

tmax_files = glob.glob(sefdir+'*Tx.tsv')
tmax_count = 0

tmin_files = glob.glob(sefdir+'*Tn.tsv')
tmin_count = 0

rain_files = glob.glob(sefdir+'*rr.tsv')
rain_count = 0


for i in range(len(mslp_files)):
    df = pd.read_csv(mslp_files[i],skiprows=12,delimiter='\t',engine='python')
    X[i] = len(df)
    mslp_count = mslp_count + len(df)

for i in range(len(tdry_files)):
    df = pd.read_csv(tdry_files[i],skiprows=12,delimiter='\t',engine='python')
    tdry_count = tdry_count + len(df)

for i in range(len(twet_files)):
    df = pd.read_csv(twet_files[i],skiprows=12,delimiter='\t',engine='python')
    twet_count = twet_count + len(df)

for i in range(len(tmax_files)):
    df = pd.read_csv(tmax_files[i],skiprows=12,delimiter='\t',engine='python')
    tmax_count = tmax_count + len(df)

for i in range(len(tmin_files)):
    df = pd.read_csv(tmin_files[i],skiprows=12,delimiter='\t',engine='python')
    tmin_count = tmin_count + len(df)

for i in range(len(rain_files)):
    df = pd.read_csv(rain_files[i],skiprows=12,delimiter='\t',engine='python')
    rain_count = rain_count + len(df)

print 'Number of mslp values = ', mslp_count
print 'Number of tdry values = ', tdry_count
print 'Number of twet values = ', twet_count
print 'Number of tmax values = ', tmax_count
print 'Number of tmin values = ', tmin_count
print 'Number of rain values = ', rain_count

total = mslp_count+tdry_count+twet_count+tmax_count+tmin_count+rain_count
print 'Total number of observations = ', total