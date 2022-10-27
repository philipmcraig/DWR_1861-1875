#import numpy as np
#import matplotlib as plt
import pylab as pl
from netCDF4 import Dataset
import xarray as xr
import glob
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from hf_funcs import *
import pcraig_funcs as pc

def MakePlot(axx,lat2D,lon2D,data,ext):
    """
    """
    axx.set_extent(ext)
    axx.coastlines(resolution='10m',zorder=4)
    borders = cfeature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land',
                                            '10m',edgecolor='k',facecolor='none') 
    axx.add_feature(borders,zorder=5)
    ct = axx.contourf(lon2D,lat2D,data,norm=pl.Normalize(100,300),cmap='viridis',
                    levels=pl.linspace(100,300,11),extend='both',zorder=2,
                                            transform=ccrs.PlateCarree())
    
    return ct

def GridLines(ax,top,left,right,bottom):
    """
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels = True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=20)
    gl.xlabels_top = top; gl.xlabels_bottom = bottom
    gl.ylabels_left = left; gl.ylabels_right = right
    gl.xlocator = mticker.FixedLocator([-12,-8,-4,0,4,8])
    gl.ylocator = mticker.FixedLocator([48,50,52,54,56,58,60,62])
    gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','size':11}
    gl.ylabel_style = {'color': 'k','size':11}
    
    return None

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR_scout461/'
raindir = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/day/v20181126'

filenames = glob.glob(raindir+'/rainfall_hadukgrid_uk_1km_day_1903*')
#filenames = filenames[:132]
#print filenames[9]

ncfile = xr.open_dataset(filenames[9])#Dataset(filenames[9],'r')
#for v in ncfile.variables:
	#print v
rain = xr.DataArray(ncfile.rainfall)#ncfile.variables['rainfall'][:]
lat = xr.DataArray(ncfile.latitude)#ncfile.variables['latitude'][:]
lon = xr.DataArray(ncfile.longitude)#ncfile.variables['longitude'][:]
#ncfile.close()

oct1903 = pl.sum(rain.values,axis=0)
rain27 = pl.sum(rain[26:28],axis=0)

x_coords = pl.arange(-200000+500,700001-500,1000)
y_coords = pl.arange(-200000+500,1250001-500,1000)

ncfile = Dataset(ncasdir+'lons_lats_1km_osgb_grid.nc','r')
lons = ncfile.variables['longitudes'][:]
lats = ncfile.variables['latitudes'][:]
ncfile.close()


#ncfile = Dataset(CR20dir+'1903/10/prate_eu_em_190310.nc','r')
#prate = ncfile.variables['PRATE_P11_L1_GGA0_avg3h'][:]
#lon2 = ncfile.variables['lon_0'][:]
#lat2 = ncfile.variables['lat_0'][:]
#ncfile.close()
ncfile = xr.open_dataset(CR20dir+'1903/10/prate_eu.nc')
prate = xr.DataArray(ncfile.PRATE_P11_L1_GGA0_avg3h)
lon2 = xr.DataArray(ncfile.lon_0)
lat2 = xr.DataArray(ncfile.lat_0)
prate = pl.mean(prate[:,1:],axis=0)

oldfile = xr.open_dataset(ncasdir+'20CR/1903/10/prate_eu_em_190310.nc')
oldpr = xr.DataArray(oldfile.PRATE_P11_L1_GGA0_avg3h)
oldpr = oldpr[1:]

pr_sum = pl.sum(prate*(60*60*3),axis=(0,1))
oldsum = pl.sum(oldpr*(60*60*3),axis=(0,1))
#pr27 = pl.zeros([4,2,lat2.size,lon2.size])#pl.sum(prate[105:109]*60*60*3,axis=(0,1))
#pr27[0,0,:,:] = prate[105,1,:,:]
pr_flat = pl.reshape(prate.values,newshape=(prate.shape[0]*prate.shape[1],
                                  prate.shape[2],prate.shape[3]))
pr27 = pl.sum(pr_flat[219:235]*60*60*3,axis=0)#219:227

#lons = pl.zeros([x_coords.size,y_coords.size])
#lats = pl.zeros([x_coords.size,y_coords.size])
#
#for i in range(x_coords.size):
#    for j in range(y_coords.size):
#        a,b = OSGB36toWGS84(x_coords[i],y_coords[j])
#        #lonlat[i,j] = (a,b)
#        lons[i,j] = a; lats[i,j] = b
#
#newnc = Dataset(jashome+'lons_lats_25km_osgb_grid.nc','w')
#
#xdim = newnc.createDimension('x',x_coords.size)
#x_in = newnc.createVariable('x',pl.float64,('x',))
#x_in.units = 'metres'
#x_in.long_name = 'metres from grid origin'
#x_in[:] = x_coords
#
#ydim = newnc.createDimension('y',y_coords.size)
#y_in = newnc.createVariable('y',pl.float64,('y',))
#y_in.units = 'metres'
#y_in.long_name = 'metres from grid origin'
#y_in[:] = y_coords
#
#lonsave = newnc.createVariable('longitudes',pl.float64,('x','y'))
#lonsave.units = 'degrees north'
#lonsave.standard_name = '60km OSGB grid longitudes'
#lonsave[:,:] = lons
#
#latsave = newnc.createVariable('latitudes',pl.float64,('x','y'))
#latsave.units = 'degrees east'
#latsave.standard_name = '60km OSGB grid latitudes'
#latsave[:,:] = lats
#
#newnc.close()

#print lons.shape
#print lats.shape
#print rain.shape
#print lat.shape
#print lon.shape

#print oct1903.shape

fig, ax = pl.subplots(1,3,figsize=(13.5,9))
proj = ccrs.Mercator()
ext = [-8,2.5,49.5,61]
#ax = pl.axes(projection=ccrs.TransverseMercator(central_longitude=-2,
#				false_easting=400000,false_northing=-100000))
#m = Basemap(projection='tmerc',llcrnrlon=-9,llcrnrlat=49,urcrnrlon=5,
#            urcrnrlat=61,resolution='i',lon_0=-2.5,lat_0=55)
#ax = pl.axes(projection=ccrs.TransverseMercator())
#ax.set_extent([-9,5,49,61],ccrs.PlateCarree())
#X, Y = m(lons,lats)
#m.drawcoastlines(); m.drawcountries()
#ax.coastlines(resolution='50m')
#cf = m.contourf(X.T,Y.T,oct1903,norm=pl.Normalize(0,300),cmap='viridis',
 #          levels=pl.linspace(0,300,13),extend='max')
#cs = pl.contourf(lons.T,lats.T,pl.flipud(oct1903),norm=pl.Normalize(0,300),
#	cmap='viridis',levels=pl.linspace(0,300,13),extend='max',
 #                   transform=ccrs.PlateCarree())
#pl.imshow(pl.flipud(oct1903),norm=pl.Normalize(0,300))
#cb = m.colorbar(cf,extend='max')
#cb.set_label('mm',fontsize=18)
#pl.title('Total UK rainfall October 1903')
#pl.savefig(ncasdir+'oct1903_total_rain_1km.png')
#pl.colorbar(cs)
#pl.show()

ax1 = pl.subplot(131,projection=proj)
#ax1.set_extent(ext)
ct = MakePlot(ax1,lats.T,lons.T,oct1903,ext)
pl.title('(a) HadUK-Grid 1km',fontsize=15)
GridLines(ax1,False,True,False,True)

ax2 = pl.subplot(132,projection=proj)
lonx,latx = pl.meshgrid(lon2,lat2)
ct = MakePlot(ax2,latx,lonx,oldsum.data,ext)
pl.title('(b) scout',fontsize=15)
ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                        edgecolor='k',
                                        facecolor='w')#cfeature.COLORS['water']
ax2.add_feature(ocean_50m,alpha=1.0,zorder=3)
GridLines(ax2,False,False,False,True)

shpfilename = shpreader.natural_earth(resolution='10m',
                                      category='cultural',
                                      name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries = reader.records()

for country in countries:
    if country.attributes['NAME_EN'] == 'Ireland':
        ax2.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor='w',zorder=10,edgecolor='k',
                          label=country.attributes['ADM0_A3'])
    elif country.attributes['NAME_EN'] == 'France':
        ax2.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor='w',zorder=10,edgecolor='k',
                          label=country.attributes['ADM0_A3'])
    elif country.attributes['NAME_EN'] == 'Belgium':
        ax2.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor='w',zorder=10,edgecolor='k',
                          label=country.attributes['ADM0_A3'])
    else:
        ax2.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor='none',
                          label=country.attributes['ADM0_A3'])



f = pl.gcf()
colax = f.add_axes([0.15,0.07,0.4,0.02])
clb = pl.colorbar(ct, cax=colax,orientation='horizontal')
clb.set_ticks(pl.linspace(100,300,11))
ticklabs = pl.asarray(pl.linspace(100,300,11))#; ticklabs = ticklabs/1000
clb.set_ticklabels(ticklabs.astype('int'))
clb.ax.tick_params(labelsize=14)
clb.update_ticks()#; cb.ax.set_aspect(0.09)
clb.set_label('mm',fontsize=18)

###############################################################################

#fig, ax = pl.subplots(1,1,figsize=(5,9))

ax3 = pl.subplot(133,projection=proj)
lonx,latx = pl.meshgrid(lon2,lat2)
ax3.set_extent(ext)
ax3.coastlines(resolution='10m',zorder=4)
borders = cfeature.NaturalEarthFeature('cultural','admin_0_boundary_lines_land',
                                        '10m',edgecolor='k',facecolor='none') 
ax3.add_feature(borders,zorder=5)
ct = ax3.contourf(lonx,latx,pr_sum-oldsum,cmap='seismic_r',norm=pl.Normalize(-20,20),
                zorder=2,extend='both',transform=ccrs.PlateCarree(),
                levels=[-20,-16,-12,-8,-4,-2,-1,1,2,4,8,12,16,20])
pl.title('(c) scout minus 20CRv3',fontsize=15)

#cb = pl.colorbar(ct,pad=0.01,orientation='horizontal')
#cb.set_ticks([-20,-16,-12,-8,-4,-2,-1,1,2,4,8,12,16,20])
#cb.set_ticklabels([-20,-16,-12,-8,-4,-2,-1,1,2,4,8,12,16,20])
#cb.ax.tick_params(labelsize=14)
#cb.update_ticks()
#cb.set_label('mm',fontsize=18)
                
f = pl.gcf()
colax = f.add_axes([0.65,0.07,0.33,0.02])
clb = pl.colorbar(ct, cax=colax,orientation='horizontal')
clb.set_ticks([-20,-16,-12,-8,-4,-2,-1,1,2,4,8,12,16,20])
ticklabs = pl.asarray([-20,-16,-12,-8,-4,-2,-1,1,2,4,8,12,16,20])#; ticklabs = ticklabs/1000
clb.set_ticklabels(ticklabs.astype('int'))
clb.ax.tick_params(labelsize=12)
clb.update_ticks()#; cb.ax.set_aspect(0.09)
clb.set_label('mm',fontsize=18)


ocean_50m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                        edgecolor='k',
                                        facecolor='w')#cfeature.COLORS['water']
ax3.add_feature(ocean_50m,alpha=1.0,zorder=3)
GridLines(ax3,False,False,True,True)

shpfilename = shpreader.natural_earth(resolution='10m',
                                      category='cultural',
                                      name='admin_0_countries')
reader = shpreader.Reader(shpfilename)
countries = reader.records()

for country in countries:
    if country.attributes['NAME_EN'] == 'Ireland':
        ax3.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor='w',zorder=10,edgecolor='k',
                          label=country.attributes['ADM0_A3'])
    elif country.attributes['NAME_EN'] == 'France':
        ax3.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor='w',zorder=10,edgecolor='k',
                          label=country.attributes['ADM0_A3'])
    elif country.attributes['NAME_EN'] == 'Belgium':
        ax3.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor='w',zorder=10,edgecolor='k',
                          label=country.attributes['ADM0_A3'])
    else:
        ax3.add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor='none',
                          label=country.attributes['ADM0_A3'])

#pl.tight_layout()
pl.subplots_adjust(left=0.025,right=0.97,wspace=0.0,top=0.96,bottom=0.12)