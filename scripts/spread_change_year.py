# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:35:41 2019

@author: pmcraig
"""

import pylab as pl
import glob
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patches as patches
import pcraig_funcs as pc

def GridLines(ax,top,left,right,bottom):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=3)
    gl.xlabels_top = top; gl.xlabels_bottom = bottom
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([-45,-30,-15,0,15,30,45])
    gl.ylocator = mticker.FixedLocator([20,40,60,80,90])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':11}
    gl.ylabel_style = {'color': 'k','size':11}
    
    return None

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'
scoutdir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR_scout461/'
#ispddir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/ISPD/'

year = '1903'
x = pl.linspace(1,12,12).astype(int)
months = ["{0:0=2d}".format(x[i-1]) for i in x]

data = pl.genfromtxt(ncasdir+'latlon_files/latlon'+year+'_pc.txt')
statlocs = data[:,1:3]

ncfile = xr.open_dataset(CR20dir+'1905'+'/'+'03'+'/prmsl.nc')
glat = xr.DataArray(ncfile.lat_0)
glon = xr.DataArray(ncfile.lon_0)
ncfile.close()

#oldsprd = pl.zeros([12,99,128])
#newsprd = pl.zeros([12,99,128])
#
#for M in range(len(months)):
#    print M
#    oldfile = xr.open_dataset(CR20dir+year+'/'+months[M]+'/prmsl_eu.nc')
#    olddata = xr.DataArray(oldfile.PRMSL_P1_L101_GGA0)
#    if M == 0:
#        lat = xr.DataArray(oldfile.lat_0)
#        lon = xr.DataArray(oldfile.lon_0)
#    #oldfile.close()
#    
#    sprd = pl.std(olddata,axis=0)
#    oldsprd[M] = pl.mean(sprd,axis=(0,1))/100
#    
#    newfile = xr.open_dataset(scoutdir+year+'/'+months[M]+'/prmsl_eu.nc')
#    newdata = xr.DataArray(newfile.PRMSL_P1_L101_GGA0)
#    #newfile.close()
#    
#    sprd = pl.std(newdata,axis=0)
#    newsprd[M] = pl.mean(sprd,axis=(0,1))/100
#    
#    oldfile.close()
#    newfile.close()


sprd_mean_chng = pl.mean(newsprd-oldsprd,axis=0)
sprd_prcnt_chng = (sprd_mean_chng/pl.mean(oldsprd,axis=0))*100

###############################################################################
fig, ax = pl.subplots(1,1,figsize=(9,6.5))
ax1 = pl.subplot(111,projection=ccrs.PlateCarree())
ax1.set_extent([-45.001,45.001,19.999,89.999],ccrs.PlateCarree())
ax1.coastlines(color='k',resolution='50m',linewidth=0.5)

X, Y = pl.meshgrid(lon,lat)
cn = ax1.contourf(X,Y,sprd_mean_chng,norm=pl.Normalize(-0.1,0.1),cmap='seismic',
	levels=[-0.1,-0.075,-0.05,-0.02,-0.01,0,0.01,0.02,0.05,0.075,0.1],extend='both',
                    alpha=0.5,transform=ccrs.PlateCarree())

ax1.contour(X,Y,sprd_mean_chng,colors='w',levels=[0],linewidths=1.5)
ct = ax1.contour(X,Y,sprd_mean_chng,colors='b',levels=[-0.4,-0.3,-0.2],
                                                linewidths=0.75,linestyles='-')
pl.clabel(ct,inline=True,fmt="%.1f",zorder=3,inline_spacing=5,manual=False,fontsize=8)

cb = pl.colorbar(cn,pad=0.025)
cb.set_ticks([-0.1,-0.075,-0.05,-0.02,-0.01,0,0.01,0.02,0.05,0.075,0.1])
cb.set_label('hPa',fontsize=15,labelpad=-1)

ax1.plot(statlocs[:,1],statlocs[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())

GridLines(ax1,True,True,False,True)

pl.tight_layout()
pl.subplots_adjust(left=0.052,right=1.02)

###############################################################################

fig, ax = pl.subplots(1,1,figsize=(9,6.5))
ax1 = pl.subplot(111,projection=ccrs.PlateCarree())
ax1.set_extent([-45.001,45.001,19.999,89.999],ccrs.PlateCarree())
ax1.coastlines(color='k',resolution='50m',linewidth=0.5)

X, Y = pl.meshgrid(lon,lat)
cn = ax1.contourf(X,Y,sprd_prcnt_chng,norm=pl.Normalize(-5,5),cmap='seismic',
	levels=[-5,-4,-3,-2,-1,0,1,2,3,4,5],extend='both',
                    alpha=0.5,transform=ccrs.PlateCarree())

ax1.contour(X,Y,sprd_prcnt_chng,colors='w',levels=[0],linewidths=1.5)
ct = ax1.contour(X,Y,sprd_prcnt_chng,colors='b',levels=[-20,-15,-10],
                                                linewidths=0.75,linestyles='-')
pl.clabel(ct,inline=True,fmt="%.0f",zorder=3,inline_spacing=5,manual=True,fontsize=8)

cb = pl.colorbar(cn,pad=0.025)
cb.set_ticks([-5,-4,-3,-2,-1,0,1,2,3,4,5])
cb.set_label('%',fontsize=15,labelpad=-1)

ax1.plot(statlocs[:,1],statlocs[:,0],marker='.',color='k',linewidth=0,
        transform=ccrs.PlateCarree())

GridLines(ax1,True,True,False,True)

rect1 = patches.Rectangle((-10.546753,49.473557),13.359254,11.228004,linewidth=1,
                      edgecolor='k',facecolor='none')
ax1.add_patch(rect1)
ax1.annotate('GBI',(-8.5,60.75),size=16)

rect2 = patches.Rectangle((-30.234253,31.929756),55.546763,35.789276,linewidth=1,
                      edgecolor='k',facecolor='none')
ax1.add_patch(rect2)
ax1.annotate('EUR',(-24.0,67.75),size=16)

pl.tight_layout()
pl.subplots_adjust(left=0.052,right=1.02)
###############################################################################

###### AREA-AVERAGED CHANGES IN SPREAD ############

lat_rad = pl.radians(glat[:])
lon_rad = pl.radians(glon[:])

lat_half = pc.HalfGrid(lat_rad)

nlon = lon_rad.shape[0] # number of longitude points
delta_lambda = (2*pl.pi)/nlon

areas = pl.zeros([lat_rad.shape[0],lon_rad.shape[0]])
radius = 6.37*(10**6)
# loop over latitude and longitude
for i in range(lat_half.shape[0]-1): # loops over 256
    latpair = (lat_half[i],lat_half[i+1])
    for j in range(lon.shape[0]): # loops over 512
        #lonpair = (lon_half[i],lon_half[i+1])
        areas[i,j] = pc.AreaFullGaussianGrid(radius,delta_lambda,latpair)

areas_eu = areas[:99,:128]

GBI_val = pl.sum((areas[41:57,49:69]*sprd_mean_chng[41:57,49:69]))/pl.sum(areas[41:57,49:69])
print round(GBI_val,2), ' hPa'

GBI_pc = pl.sum((areas[41:57,49:69]*sprd_prcnt_chng[41:57,49:69]))/pl.sum(areas[41:57,49:69])
print round(GBI_pc,2), ' %'

EU_val = pl.sum((areas_eu[31:83,21:101]*sprd_mean_chng[31:83,21:101]))/pl.sum(areas_eu[31:837,21:101])
print round(EU_val,2), ' hPa'

EU_pc = pl.sum((areas_eu[31:83,21:101]*sprd_prcnt_chng[31:83,21:101]))/pl.sum(areas_eu[31:837,21:101])
print round(EU_pc,2), ' %'