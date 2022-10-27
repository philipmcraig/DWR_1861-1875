# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:16:30 2019

@author: pmcraig
"""

import pylab as pl
from netCDF4 import Dataset
import glob
from mpl_toolkits.basemap import Basemap, shiftgrid
import timeit
from scipy.stats import skew
from matplotlib.colors import Normalize
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox
from matplotlib.patches import Rectangle
import pcraig_funcs as pc


class MidpointNormalize(Normalize):
    """Found this on the internet. Use to centre colour bar at zero.
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        a, b = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return pl.ma.masked_array(pl.interp(value, a, b))

def HistFunc(var,point,jashome,statname,presob,savetime):#,i,j):
    """
    """
    varstat = pl.zeros([var.shape[0]])
    for i in range(var.shape[0]):
        varstat[i] = pc.BilinInterp(point,lon,lat,var[i])
    
    mn = pl.mean(varstat)
    sd = pl.std(varstat)
    err = (mn-presob*100)/sd
    
    fig = pl.figure(); ax = pl.gca()
    n, bins, patches = pl.hist(varstat,20,ec='k',normed=1,zorder=2)
    y = pl.normpdf(bins,mn,sd)
    pl.plot(bins,y)
    sk = skew(varstat)
    ff = 'figure fraction'
    
    skwtxt = 'skewness = '+str(round(sk,2))
    errtxt = 'error = '+str(round(err,2))
    texts = [skwtxt,errtxt]
    Texts = []
    for t in texts:
        Texts.append(TextArea(t))
    texts_vbox = VPacker(children=Texts,pad=0,sep=0)
    ann = AnnotationBbox(texts_vbox,xy=(.17,.85),xycoords=ff,box_alignment=(0,.5),
                         bboxprops=dict(color='2',alpha=1,zorder=4),pad=0.25)
    ax.add_patch(Rectangle((.16,.81),.21,.08,fc='none',ec='k',lw=1.5,zorder=5,
                           transform=fig.transFigure))
    ann.set_figure(fig)
    fig.artists.append(ann)
    #pl.annotate('skewness = '+str(round(sk,2)),xy=(0.17,0.85),xycoords=ff,
    #            zorder=4,bbox=dict(facecolor='white', edgecolor='k', pad=10.0))
    #pl.annotate('error = '+str(round(err,2)),xy=(0.17,0.8),xycoords=ff,zorder=4)
    pl.title('ensemble spread at '+statname)
    pl.xlabel('pressure (Pa)',fontsize=15)
    pl.ylabel('probability density',fontsize=15)
    pl.subplots_adjust(left=0.15,right=0.98,top=0.94)
    pl.grid(axis='y',ls='--',zorder=0,color='lightgrey')
    #n = str(i) + str(j)
    saveyr = savetime[:4]
    pl.savefig(jashome+'mslp_spread_'+saveyr+'/stations/hist_'+savetime+'_'+
                                statname.replace(' ','')+'.png')
    
    return sk

pl.close('all')

jashome = '/home/users/pmcraig/'
ncasdir = '/gws/nopw/j04/ncas_climate_vol1/users/pmcraig/'
raindir = '/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.0.0.0/1km/rainfall/day/v20181126/'
CR20dir = '/gws/nopw/j04/ncas_climate_vol1/users/ehawkins/20CR/'

data = pl.genfromtxt(ncasdir+'latlon_files/latlon1905_pc.txt')
stats = pl.where((data[:,-4]==8) & (data[:,-3]==0))
statdata = data[stats[0][:-1]]

obs = pl.genfromtxt(ncasdir+'DWRcsv/1905/dwr_1905_03_15.csv',delimiter=',')
pres = obs[stats[0][:-1],3]
#correct for gravity and change to hPa:
pres[0] = pres[0]+0.04 # Sumburgh Head
N = [1,2,15,16,17,18,19] # Malin Head/Shields and north (not Sumburgh Head)
pres[N] = pres[N] + 0.03
S = [3,4,5,6,7,8,9,10,13,14,20,21,22,23,24,25,26,27] # south of Malin/Shields
pres[S] = pres[S] + 0.02
pres[[11,12]] = pres[[11,12]] + 0.01 # Scilly & Jersey
pres = pres*33.8638

locs = statdata[:,1:3]


B = pl.genfromtxt(ncasdir+'DWRcsv/1905/dwr_1905_03_15.csv',delimiter=',',dtype=object)
names = B[:,0]; del B
names = names[stats[0][:-1]]

start_time = timeit.default_timer()
presfile = Dataset(CR20dir + '1905/03/prmsl_eu.nc','r')
mslp = presfile.variables['PRMSL_P1_L101_GGA0'][:]
lat = presfile.variables['lat_0'][:]
lon = presfile.variables['lon_0'][:]
inittime = presfile.variables['initial_time0'][:]
foretime = presfile.variables['forecast_time0'][:]
presfile.close()
elapsed = timeit.default_timer() - start_time
print elapsed/60

new = pl.zeros_like(mslp)
new[:,:,:,:,:mslp.shape[-1]/2] = mslp[:,:,:,:,mslp.shape[-1]/2:]
new[:,:,:,:,mslp.shape[-1]/2:] = mslp[:,:,:,:,:mslp.shape[-1]/2]
mslp = new.copy(); del new

std = pl.std(mslp,axis=0)

#new = pl.zeros_like(std)
#new[:,:,:,:std.shape[-1]/2] = std[:,:,:,std.shape[-1]/2:]
#new[:,:,:,std.shape[-1]/2:] = std[:,:,:,:std.shape[-1]/2]
#std = new.copy(); del new

newlon = pl.zeros_like(lon)
newlon[:lon.size/2] = lon[lon.size/2:] - 360
newlon[lon.size/2:] = lon[:lon.size/2]
lon = newlon.copy(); del newlon

#lat1 = 58.727; lat2 = 54.495
#lon1 = -8.410; lon2 = -0.636
#
#upper = pc.NearestIndex(lat,lat1); lower = pc.NearestIndex(lat,lat2)
#west = pc.NearestIndex(lon,lon1); east = pc.NearestIndex(lon,lon2)
#p1 = (lon[west],lat[upper]); p2 = (lon[east],lat[upper])
#p3 = (lon[east],lat[lower]); p4 = (lon[west],lat[lower])

#for i in enumerate(pl.linspace(-12,-1,12)):
#    for frc in range(0,2):
pl.figure(figsize=(10,6))#2*i[0]+frc,
m = Basemap(projection='cyl',llcrnrlon=-10,urcrnrlon=3,llcrnrlat=50,
            urcrnrlat=60,resolution='i')
m.drawcoastlines(color='k',linewidth=1,zorder=4)

#m.plot((p1[0],p2[0]),(p1[1],p2[1]),marker=None,ls='-',color='g',latlon=True)
#m.plot((p2[0],p3[0]),(p2[1],p3[1]),marker=None,ls='-',color='g',latlon=True)
#m.plot((p3[0],p4[0]),(p3[1],p4[1]),marker=None,ls='-',color='g',latlon=True)
#m.plot((p4[0],p1[0]),(p4[1],p1[1]),marker=None,ls='-',color='g',latlon=True)

ind = 57 # which index in zero-axis of mslp_em to use
frc = 0 # 0 or 1, which forecast time to use (+ 0 hours or + 3 hours)
#inittime[ind,-5] = str(int(inittime[ind,-5])+foretime[frc])
hour = str("{0:0=2d}".format(int(''.join(inittime[ind,-6:-4]))+foretime[frc]+2))
inittime[ind,-6] = hour[0]; inittime[ind,-5] = hour[1]
time = ''.join([str(x) for x in inittime[ind]]) # combine strings into one
print time

M = mslp[:,ind,frc]
M2 = mslp[:,ind,frc+1]
dp_dt = (M2-M)/(3*60*60)
MI = M + dp_dt*(2*60*60)

lons,lats = pl.meshgrid(lon,lat)
X, Y = m(lons,lats)
levels = pl.arange(0,13,2)
#ct = m.contourf(X,Y,M,levels=levels,zorder=2,extend='max')
#cb = m.colorbar(ct,location='right')
#cb.set_label('hPa',fontsize=14)

m.drawmeridians([-20,-10,0,10,20],linewidth=0.5,dashes=[1,2],zorder=3,
                color='grey',labels=[1,1,0,1],fontsize=15)
m.drawparallels([40,50,60,70],linewidth=0.5,dashes=[1,2],zorder=3,color='grey',
                labels=[1,0,0,0],fontsize=15)

#pl.title('mslp ensemble $\sigma$ '+time)
pl.tight_layout()
pl.subplots_adjust(bottom=0.05,left=0.06,right=0.94)
year = time[6:10]; month = time[0:2]; day = time[3:5]
step = time[-6:-4]+time[-3:-1]
savetime = year+month+day+'_'+step#+'_'
#pl.savefig(jashome+'mslp_spread_1903/mslp_std_'+savetime+'.png')

#var = mslp[:,ind,frc,upper:lower+1,west:east+1]#[:,3,0]
#lon_sub = lon[west:east+1]; lat_sub = lat[upper:lower+1]
#S = pl.zeros([lat_sub.size,lon_sub.size])

for i in range(locs.shape[0]):
    #for j in range(var.shape[2]):
    #pl.figure()
    if i==22:
        pass
    else: #mslp[:,ind,frc]
        sk = HistFunc(MI,(locs[i,1],locs[i,0]),jashome,names[i],
                      pres[i],savetime)#,i,j)
#    S[i,j] = sk
#
#pl.figure()
#lons,lats = pl.meshgrid(lon_sub,lat_sub)
#X, Y = m(lons,lats)
#m.drawcoastlines()
#m.contourf(X,Y,S,cmap='RdYlGn',levels=pl.arange(-0.7,0.71,0.2),extend='min')
#m.colorbar()