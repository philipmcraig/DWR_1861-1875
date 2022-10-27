# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:35:55 2019

@author: pmcraig
"""

import pylab as pl
from netCDF4 import Dataset
from scipy.stats import skew
import xarray as xr
import timeit
import glob
import os

def MainFunc(CR20dir,ncasdir,year,month,vartype):
    """
    Args:
        CR20dir (string): Ed's JASMIN directory with 20CR netcdf files
        ncasdir (string): my JASMIN directory
        year (string): year
        month(string): month
        vartype (string): filenames for 20CR netcdf files (prmsl, prate, air.2m)
    """
    # open ncfile:
    ncfile = xr.open_dataset(CR20dir+year+'/'+month+'/'+vartype+'_eu.nc')
    
    # extract data:
    # need to have if statements for mslp, precip & temp
    if vartype == 'prmsl':
        data = xr.DataArray(ncfile.PRMSL_P1_L101_GGA0)
        newvname = 'mslp'
    elif vartype == 'prate':
        data = xr.DataArray(ncfile.PRATE_P11_L1_GGA0_avg3h)
        newvname = 'precipitation'
    elif vartype == 'air.2m':
        data = xr.DataArray(ncfile.TMP_P1_L103_GGA0)
        newvname = 'temperature'

    # set up empty array for skewness, sans zero axis:
    sk = pl.zeros([data.shape[1],data.shape[2],data.shape[3],data.shape[4]])
    
    # get the dimensions:
    D = list(data.dims)
    D.remove(D[0]) # remove ensemble members dimension
    
    # make skewness array a DataArray
    sk = xr.DataArray(sk,dims=D)

    for i in range(data.shape[2]): # loop over forecast times
        for j in range(data.shape[3]): # loop over latitude
            for k in range(data.shape[4]): # loop over longitude
                # calculate skewness along initial times axis:
                sk[:,i,j,k] = skew(data[:,:,i,j,k],axis=0)

    # get the co-ordinates:
    C = dict(data.coords)
    C.pop('ensemble0') # remove ensemble members co-ordinate(s)
    
    # make skewness DataArray into a Dataset:
    # need to have if statemenes for mslp, precipt & temp
    ds = xr.Dataset({newvname+' skewness': sk},coords=C)
    
    # save Dataset to netcdf file:
    ds.to_netcdf(ncasdir+'20CR/'+year+'/'+month+'/'+newvname+
                                            '_eu_skew_'+year+month+'.nc')
    
    return None

  
jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
years = pl.linspace(1900,1910,11).astype(int)#'1906'
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

VARTYPE = 2 # mslp (0), precip (1), temp (2)
# var strings are the 20CR filenames 
if VARTYPE == 0:
    print 'calculating skewness of mslp'
    var = 'prmsl'
elif VARTYPE == 1:
    print 'calculating skewness of precip'
    var = 'prate'
elif VARTYPE == 2:
    print 'calculating skewness of temperature'
    var = 'air.2m'

for Y in range(len(years)):
    for M in range(len(months)):# loop over months
        yr = str(years[Y])
        mn = months[M]
        exists = os.path.isfile(CR20dir+str(yr)+'/'+mn+'/'+var+'_eu.nc')
        
        if exists == True:
            print var+'.nc exists: calculating skewness for '+yr+' '+mn
            MainFunc(CR20dir,ncasdir,yr,mn,var)
        elif exists == False:
            print var+'.nc '+yr+' '+mn+'does not exist: move on to next file'

#A = xr.open_dataset(CR20dir+year+'/08/prmsl_eu.nc')
#
#start_time = timeit.default_timer()
#mslp = xr.DataArray(A.PRMSL_P1_L101_GGA0)
#elapsed = timeit.default_timer() - start_time
#print elapsed/60
#
##start_time = timeit.default_timer()
#sk = pl.zeros([mslp.shape[1],mslp.shape[2],mslp.shape[3],mslp.shape[4]])
#D = list(mslp.dims)
#D.remove(D[0])
#sk = xr.DataArray(sk,dims=D)
#for i in range(mslp.shape[1]):
#for j in range(mslp.shape[2]):
#    for k in range(mslp.shape[3]):
#        for l in range(mslp.shape[4]):
#            sk[:,j,k,l] = skew(mslp[:,:,j,k,l],axis=0)
#elapsed = timeit.default_timer() - start_time
#print elapsed/60
#
#C = dict(mslp.coords)
#C.pop('ensemble0')
#
#ds = xr.Dataset({'mslp skewness': sk},coords=C)

#ds.to_netcdf(ncasdir+'20CR/'+year+'/08/prmsl_eu_skew_'+year+'08'+'.nc')