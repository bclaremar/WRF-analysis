#!/bin/python3

from __future__ import print_function
import sys,getopt 
#from netCDF4 import Dataset # use scipy instead 
from scipy.io import netcdf 
import matplotlib 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import numpy as np 
import os 
from datetime import datetime, timedelta 
from calendar import monthrange
#import coltbls as coltbls 
import wrf
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords, ALL_TIMES , xy_to_ll, ll_to_xy, interplevel,) 
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.axes as maxes
import xarray 
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy

import convpar

#def main():
#    pid=os.fork()
#    if pid==0:
#        showFigure()


dos='2'
ms='10'
ds='14'
hs='18'
hst='9'
ys='2023'

ms,ds, hs, dos, steps = input("").split()

m0=int(ms)
d0=int(ds)
do=int(dos)
h0=int(hs)
step=int(steps)
ncfile=netcdf.netcdf_file("wrfout_d0" + dos + "_2023-" + ms + "-" + ds + "_" + hs + ":00:00",'r')

time = getvar(ncfile, "XTIME",ALL_TIMES)/60+h0+1
print(time)
print(do)
#if do==2:
#    ta=np.where(time==ht)
#    if ht<h0:
#        ta=np.where(np.mod((time),24)==ht)
#else:
#    ta=np.where(np.floor(time)==ht)
#    if ht<h0:
#        ta=np.where(np.mod(np.floor(time),24)==ht)

    
t=int(step)
print(t)
print(time[t])
hh=str(int(np.mod(time[t],24)))
print(hh)

dst=ds
mst=ms
dt=np.copy(d0)
mt=np.copy(m0)
print(dt)
print(mt)
if time[t]>=24:
    dt=d0+1
    dst=str(dt)
    if dt<10:
      dst='0'+str(dt)
dx=monthrange(2023, m0)
if dt>dx[1]:
    dt=1
    dst="0" + str(dt)
    mt=m0+1
    mst="0" + str(mt)
    
datestring=(ys + "-" + mst + "-" + dst)
print(datestring)

#exit()

slp = getvar(ncfile, "slp",timeidx=t)
T2 = getvar(ncfile, "T2",timeidx=t)-273.15
q2 = getvar(ncfile, "Q2",timeidx=t)
TD2 = wrf.td(slp, q2, meta=True, units='degC')
RH2=wrf.rh(q2, slp*100, T2, meta=True)
u10 = getvar(ncfile, "U10",timeidx=t)
v10 = getvar(ncfile, "V10",timeidx=t)
U10 = np.sqrt(u10**2 +v10**2)
mdbz = getvar(ncfile, "mdbz",timeidx=t)
R=np.power(np.power(10,mdbz/10)/200,0.625)
hel = getvar(ncfile, "helicity",timeidx=t)

rains1 = getvar(ncfile, "RAINNC",timeidx=t-1)
rains2 = getvar(ncfile, "RAINNC",timeidx=t)
rainc1 = getvar(ncfile, "RAINC",timeidx=t-1)
rainc2= getvar(ncfile, "RAINC",timeidx=t)
rains=rains2-rains1
rainc=rainc2-rainc1
rain=rains+rainc
[MCAPE, MCIN, LCL, LFC]=getvar(ncfile, "cape_2d",timeidx=t)
CTT=getvar(ncfile, "ctt",timeidx=t)


Q = getvar(ncfile, "QVAPOR",timeidx=t)*1000
TH = getvar(ncfile, "th",timeidx=t)
P = getvar(ncfile, "pressure",timeidx=t)
T = getvar(ncfile,"tk",timeidx=t) 
wspd =  getvar(ncfile, "wspd_wdir",timeidx=t, units="mps")[0,:]

z = getvar(ncfile, "z")
RH=wrf.rh(Q/1000, P*100, T, meta=True)
EPT=wrf.eth(Q/1000, T, P*100, meta=True, units='K')-273.15
QS=np.divide(Q,RH/100)
EPTS=wrf.eth(QS/1000, T, P*100, meta=True, units='K')-273.15


ht = getvar(ncfile, "z",timeidx=t, units="m")
ht_1000=wrf.vinterp(ncfile,ht,'p',[1000.0, 500.0],field_type='ght',extrapolate=True,timeidx=t)
#hg_1000 = ncfile.variables['Geopotential_height_isobaric'][0, plev.index(1000)]

#ht_1000 = interplevel(ht, P, 1000.0,extrapolate=True)
ht_500 = interplevel(ht, P, 500.0)
t_500 = interplevel(T, P, 500.0)-273.15
ht_850 = interplevel(ht, P, 850.0)
t_850 = interplevel(T, P, 850.0)-273.15
ept_850 = interplevel(EPT,P,850.0)
ht_700 = interplevel(ht, P, 700.0)
rh_700 = interplevel(RH,P,700.0)
t_700 = interplevel(T,P,700.0)-273.15
ht_300 = interplevel(ht, P, 300.0)
rh_300 = interplevel(RH,P,300.0)
ws_300 = interplevel(wspd,P,300.0)
t_300 = interplevel(T,P,300.0)-273.15

smooth_slp = smooth2d(slp, 3, cenweight=4)
thick=ht_500-ht_1000[0, : , : ] 

# Get the latitude and longitude points
lats, lons = latlon_coords(slp)

# Get the cartopy mapping object
cart_proj = get_cartopy(slp)

#Calc LFC,LEQ,TEQ
EPTS=xarray.DataArray.to_numpy(EPTS)
EPT=xarray.DataArray.to_numpy(EPT)
Thae=EPT[0,:,:]
Tt=T-273.15
size=np.shape(EPT)
print(size)
size=np.shape(Thae)
print(size)

LEQ=np.full_like(T2,0)
TEQ=np.full_like(T2,np.nan)
LFC=np.full_like(T2,np.nan)
LEQ2=np.full_like(T2,0)
LFC2=np.full_like(T2,np.nan)
zp2=np.sqrt(z)

a=np.where(EPTS-Thae<0)
lOC= list(zip(a[1], a[2], a[0]))
b=np.array(lOC)
lb=len(b)
kx=np.max(b[:,2])
print(kx)

'''
a=np.where(EPTS-Thae<0)
lOC= list(zip(a[1], a[2], a[0]))
b=np.array(lOC)
print(len(b))
I=0
c=np.full_like(b,0)
c[I,:3]=b[0,:]
for k in range(1,len(b)):
#    c[b[k,0],b[k,1]]=b[k,2]
#    print(k,I)
    c[I,:3]=b[k,:]
    if np.not_equal(b[k,:1],b[k-1,:1]):
        I=I+1 
d=c[:I,:]        
print(len(d))
b=d
for k in range(len(b)):
    if k%1000==0:
         print(k)
    LEQ[b[k,0],b[k,1]]=z[b[k,2],b[k,0],b[k,1]]
    TEQ[b[k,0],b[k,1]]=Tt[b[k,2],b[k,0],b[k,1]]

#LEQ[b[:,0],b[:,1]]=z[b[:,2],b[:,0],b[:,1]]
#TEQ[b[:,0],b[:,1]]=Tt[b[:,2],b[:,0],b[:,1]]

'''

'''
for i in np.arange(size[0]):
    print(i)
    for j in np.arange(size[1]):
        #print(type(j),type(pi),type(LFCp))
        LFC[i,j], LEQ[i,j], TEQ[i,j],LFC2[i,j], LEQ2[i,j] = convpar.convpar(Thae[i,j],EPTS[:,i,j],z[:,i,j],zp2[:,i,j],Tt[:,i,j])
'''

kkx=kx
for k in range(len(b)-1,0,-1):
    kx=b[k,2]
    i=b[k,0]
    j=b[k,1]
    if kx-kkx:
    	print(kx)
    if np.isnan(TEQ[i,j]):
        TEQ[i,j]=Tt[kx,i,j]     
    kkx=kx

#print(LFC)
print(LEQ)
print(TEQ)

bx=10;

# Create a figure
fig = plt.figure(0,figsize=(16,6))
ax=fig.add_subplot(1,2,1,projection=cart_proj)
#Set the GeoAxes to the projection used by WRF
#ax = plt.axes(projection=cart_proj)
# Download and add the states and coastlines
#ax.add_feature(cfeature.LAND)
#ax.add_feature(cfeature.OCEAN)
#land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
#                                        edgecolor='face',
#                                        facecolor=cfeature.COLORS['land'])
                                        
#states = NaturalEarthFeature(category="cultural", scale="50m",
#                             facecolor="none",
#                             name="admin_1_states_provinces_shp")
#ax.add_feature(states, linewidth=.5, edgecolor="black")
#ax.coastlines('50m', linewidth=0.8)
#rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '50m')

#ax.add_feature(rivers_50m, facecolor='None', edgecolor='b')
# Make the contour outlines and filled contours for the smoothed sea level
# pressure.
p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="black",
            transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
ht=plt.contour(to_np(lons), to_np(lats), to_np(T2), range(-50,50,5), colors="red",
            transform=crs.PlateCarree())
T2_contours =plt.contourf(to_np(lons),to_np(lats),to_np(T2),levels=range(-26,14+1,2),
             transform=crs.PlateCarree(),
             cmap=get_cmap("hsv_r"))
plt.colorbar(T2_contours, ax=ax)
ax.set_title('T2 at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")

# Create a figure
ax=fig.add_subplot(1,2,2,projection=cart_proj)

p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="black",    transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
ht=plt.contour(to_np(lons), to_np(lats), to_np(TD2), range(-30,50,5), colors="red",
            transform=crs.PlateCarree())
TD2_contours =plt.contourf(to_np(lons),to_np(lats),to_np(TD2),levels=range(-26,14+1,2),
            transform=crs.PlateCarree(),
             cmap=get_cmap("hsv_r"))
plt.colorbar(TD2_contours, ax=ax)
ax.set_title('TD2 at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")

if do==1:
  file='maps/map_'+steps+'.png'
  print(file)
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='maps/map_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')



# Create a figure
fig = plt.figure(1,figsize=(16,6))
ax=fig.add_subplot(1,2,1,projection=cart_proj)
#ax = plt.axes(projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
lats, lons = latlon_coords(U10)

p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="black",    transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
U10_contours =plt.contourf(to_np(lons),to_np(lats),to_np(U10), 			levels=range(0,25,2),
             transform=crs.PlateCarree(),
             cmap=get_cmap("jet"))
plt.colorbar(U10_contours, ax=ax)
plt.barbs(to_np(lons[::bx,::bx]), to_np(lats[::bx,::bx]),
          to_np(u10[::bx, ::bx]), to_np(v10[::bx, ::bx]),
          transform=crs.PlateCarree(), length=6)
ax.set_title('U10 at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")

# Create a figure
ax=fig.add_subplot(1,2,2,projection=cart_proj)
#Set the GeoAxes to the projection used by WRF
#ax = plt.axes(projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)

ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)

p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="black",    transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
#plt.contour(to_np(lons), to_np(lats), to_np(rains), levels=[0.5,1,2,4,10], colors="black",    transform=crs.PlateCarree())
plt.contour(to_np(lons), to_np(lats), to_np(rainc), levels=[0.5,1,2,4,10], colors="red",    transform=crs.PlateCarree())
rain_contours =plt.contourf(to_np(lons),to_np(lats),to_np(np.log10(rain)), 			levels=np.log10([0.1,0.2,0.5,1,2,5,10,20]), extend="max",
             transform=crs.PlateCarree(),
             cmap=get_cmap("jet"))
cbar=plt.colorbar(rain_contours, ax=ax)
cbar.ax.set_yticklabels([.1,.2,.5,1,2,5,10,20])

if do==1:
	ax.set_title('2-hour precip until ' + hh + ":00 SNT " + datestring)
else:
	ax.set_title('20-min precip until ' + hh + ":00 SNT " + datestring)

	
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)


if do==1:
  file='maps/map12_'+steps+'.png'
  print(file)
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='maps/map12_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')


fig = plt.figure(2,figsize=(16,6))
ax=fig.add_subplot(1,2,1,projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="white",
            transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
p=plt.contour(to_np(lons),to_np(lats),ht_500,levels=range(4800,5761,60), colors="black",
             transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_500, [-45, -10, -5, 0], colors="red",
            transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_500, range(-50,10,2), colors="red", linewidth=2, linestyle=":",transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
thick_contours =plt.contourf(to_np(lons),to_np(lats),thick, 			levels=range(4800,5760,40),
             transform=crs.PlateCarree(),
             cmap=get_cmap('hsv_r'))
#'nipy_spectral'
plt.colorbar(thick_contours, ax=ax)
tk=plt.contour(to_np(lons),to_np(lats),thick,                     levels=range(4800,5760,40),colors="red",
             transform=crs.PlateCarree())
ax.clabel(tk, tk.levels, inline=True, fontsize=10)
ax.set_title('Thick500 at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(color="black", linestyle="dotted")

ax=fig.add_subplot(1,2,2,projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
ht=plt.contour(to_np(lons), to_np(lats), ht_850, range(900,1800,30), colors="black",
            transform=crs.PlateCarree())
ax.clabel(ht, ht.levels, inline=True, fontsize=10)
ht=plt.contour(to_np(lons), to_np(lats), ept_850, [0, 15, 30, 45, 60], colors="black",
            transform=crs.PlateCarree())
ax.clabel(ht, ht.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_850, [-15, -10, -5, 0, 5, 15 ], colors="red",
            transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_850, range(-20,21,2), colors="red", linewidth=2, linestyle=":",transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
ept850_contours =plt.contourf(to_np(lons),to_np(lats),ept_850,range(-15,57+1,3),
             transform=crs.PlateCarree(),
             cmap=get_cmap("hsv_r"))
plt.colorbar(ept850_contours, ax=ax)
ax.set_title('EPT850 at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")
if do==1:
  file='maps/map2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='maps/map2_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')


fig = plt.figure(3,figsize=(16,6))
ax=fig.add_subplot(1,2,1,projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
ht=plt.contour(to_np(lons), to_np(lats), ht_700, 10, colors="black",
            transform=crs.PlateCarree())
ax.clabel(ht, ht.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_700, range(-30,10,2), colors="red", linewidth=2, linestyle=":",transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
rh700_contours =plt.contourf(to_np(lons),to_np(lats),rh_700, 			10,
             transform=crs.PlateCarree(),
             cmap=get_cmap("YlGn"))
plt.colorbar(rh700_contours, ax=ax)
ax.set_title('RH700 at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")

ax=fig.add_subplot(1,2,2,projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
ht=plt.contour(to_np(lons), to_np(lats), ht_300, range(8400,9900,60), colors="black",
            transform=crs.PlateCarree())
ax.clabel(ht, ht.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_300, range(-70,35,2), colors="red", linewidth=2, linestyle=":",transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
ws300_contours =plt.contourf(to_np(lons),to_np(lats),ws_300, range(0,100,5),
             transform=crs.PlateCarree(),
             cmap=get_cmap("jet"))
plt.colorbar(ws300_contours, ax=ax)
ws=plt.contour(to_np(lons), to_np(lats), ws_300, levels=[20,40,60, 80,100], colors="red",
            transform=crs.PlateCarree())
ax.clabel(ws, ws.levels, inline=True, fontsize=10)
ax.set_title('WS300 at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
# Add the gridlines
ax.gridlines(color="black", linestyle="dotted")
if do==1:
  file='maps/map22_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='maps/map22_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')


fig = plt.figure(4,figsize=(16,6))
ax=fig.add_subplot(1,2,1,projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="black",
            transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_500, range(-50,10,2), colors="red", linewidth=2, linestyle=":",transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
mcape=plt.contour(to_np(lons), to_np(lats), MCAPE, levels=[10,50,100,200,500,1000], linewidth=2, linestyle="-",transform=crs.PlateCarree())
try:
  ax.clabel(mcape, mcape.levels, inline=True, fontsize=10)
except:
  print('no mcape levels')
  
teq=plt.contour(to_np(lons), to_np(lats), TEQ[:,:], range(-50,11,10), colors="green", linewidth=2, linestyle="-",transform=crs.PlateCarree())
teq_contours =plt.contourf(to_np(lons),to_np(lats),TEQ, range(-50,11,5), transform=crs.PlateCarree())
#'nipy_spectral'
plt.colorbar(teq_contours, ax=ax)
#tk=plt.contour(to_np(lons),to_np(lats),MCIN,range(50,101,50),colors="blue",  transform=crs.PlateCarree())
#ax.clabel(tk, tk.levels, inline=True, fontsize=10)
ax.set_title('TEQ + MCAPE at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(color="black", linestyle="dotted")

ax=fig.add_subplot(1,2,2,projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="black",
            transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_500, range(-50,10,2), colors="blue", linewidth=2, linestyle=":",transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
thick_contours =plt.contourf(to_np(lons),to_np(lats),CTT, range(-70,10,10),cmap=get_cmap("hsv_r"), transform=crs.PlateCarree())
#'nipy_spectral'
plt.colorbar(thick_contours, ax=ax)
tk=plt.contour(to_np(lons),to_np(lats),MCAPE,colors="red", levels=[50,100,300,1000], transform=crs.PlateCarree())
try:
  ax.clabel(tk, tk.levels, inline=True, fontsize=10)
except:
  print('no tk levels')
ax.set_title('CTT+MCAPE at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(color="black", linestyle="dotted")
if do==1:
  file='maps/map3_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='maps/map3_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')

fig = plt.figure(5)
ax=fig.add_subplot(1,1,1,projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="black",
            transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_500, range(-50,10,2), colors="blue", linewidth=2, linestyle=":",transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
thick_contours =plt.contourf(to_np(lons),to_np(lats),mdbz, levels=range(0,61,5), transform=crs.PlateCarree(),cmap=get_cmap("jet"))
#'nipy_spectral'
plt.colorbar(thick_contours, ax=ax)
tk=plt.contour(to_np(lons),to_np(lats),hel,colors="red", transform=crs.PlateCarree())
ax.clabel(tk, tk.levels, inline=True, fontsize=10)
ax.set_title('max dBz at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(color="black", linestyle="dotted")

fig = plt.figure(6)
ax=fig.add_subplot(1,1,1,projection=cart_proj)
#ax.coastlines('50m', linewidth=0.8)
p=plt.contour(to_np(lons), to_np(lats), to_np(smooth_slp), levels=range(960,1061,4), colors="black",
            transform=crs.PlateCarree())
ax.clabel(p, p.levels, inline=True, fontsize=10)
tt=plt.contour(to_np(lons), to_np(lats), t_500, range(-50,10,2), colors="blue", linewidth=2, linestyle=":",transform=crs.PlateCarree())
ax.clabel(tt, tt.levels, inline=True, fontsize=10)
radar =plt.contourf(to_np(lons),to_np(lats),mdbz, levels=[7,12,18,23,28,34,39,44,50,55],  transform=crs.PlateCarree(),cmap=get_cmap("jet"))
#'nipy_spectral'
cbar=plt.colorbar(radar, ax=ax)
#cbar = plt.colorbar(radar)
cbar.ax.set_yticklabels([.1,.2,.5,1,2,5,10,20,50,100])

tk=plt.contour(to_np(lons),to_np(lats),hel,colors="red",levels=[100,250], transform=crs.PlateCarree())
ax.clabel(tk, tk.levels, inline=True, fontsize=10)
ax.set_title('max precip/h + helicity at ' + hh + ":00 SNT " + datestring)
# Set the map bounds
ax.set_xlim(cartopy_xlim(smooth_slp))
ax.set_ylim(cartopy_ylim(smooth_slp))
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.4)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(color="black", linestyle="dotted")
if do==1:
  file='maps/map4_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='maps/map4_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')

pid=os.fork()
if pid==0:
  plt.show()


