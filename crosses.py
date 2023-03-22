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
import xarray as xr
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy

def Td_ekvT(T,r,p):
    g=9.82
    Rd=287.06
    TK=T+273.15  #Temp i Kelvin
    #print (np.log(abs(TK)))
    E=np.power(10,(24.00978-2957.07/TK-2.20999*np.log(abs(TK))))*r /100  #angtryck i hPa
    Q=.62198*E/(p-.37802*E)  #specifik fuktighet
    TV=(1+.61*Q)*TK  #Virtuell temp i Kelvin
    TD=TK/(1+287.06*TK/0.62198/(2.5008e6-2348.4*T)*np.log(100/r))-273.15  #Daggpunkt i C

    Daggpunkt= np.round(TD, 1)
    specfukt= np.round(Q*10000)/10
    print('Daggpunkt = ' + str(Daggpunkt))
    print('Q = ' + str(specfukt))

    Theta=TK*np.power(1000/p,2/7)-273.15#potentiell temp i C
    cp=1004.71#specifik varmekapacitet
    S=(6*(273.15+TD)-TK)/5#Mattnadspunkt i K
    L=2.5008e6-2348.4*(S-273.15)#Latent varme
    Tae=TK*np.exp(L*Q/cp/S)#Ekvivalenttemperatur i K
    thae=Tae*np.power(1000/p,2/7)-273.15#potentiell ekvivalenttemp i C

    Thae= np.round(thae, 1)
    print('Thae = ' + str(Thae))
    return(TD,Thae,specfukt)





ys='2023'
dos='2'
ms='10'
ds='14'
hs='18'
hst='9'
ms,ds, hs, dos, steps = input("").split()
do=int(dos)
m0=int(ms)
d0=int(ds)
h0=int(hs)
step=int(steps)
#ht=int(hst)
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

print(step)
print(steps)
t=int(step)
    
#print(ta)
#t=int(ta[0])
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
dx=monthrange(2023, m0)
if dt>dx[1]:
    dt=1
    dst="0" + str(dt)
    mt=m0+1
    mst="0" + str(mt)
    
datestring=(ys + "-" + mst + "-" + dst)
print(datestring)


slp = getvar(ncfile, "slp",timeidx=t)
T2 = getvar(ncfile, "T2",timeidx=t)-273.15
q2 = getvar(ncfile, "Q2",timeidx=t)
TD2 = wrf.td(slp, q2, meta=True, units='degC')
RH2=wrf.rh(q2, slp*100, T2, meta=True)
u10 = getvar(ncfile, "U10",timeidx=t)
v10 = getvar(ncfile, "V10",timeidx=t)
U10 = np.sqrt(u10**2 +v10**2)

m,n=T2.shape
print(m,n)

Q = getvar(ncfile, "QVAPOR",timeidx=t)*1000
TH = getvar(ncfile, "th",timeidx=t)
P = getvar(ncfile, "pressure",timeidx=t)
T = getvar(ncfile,"tk",timeidx=t) 
wspd =  getvar(ncfile, "wspd_wdir",timeidx=t, units="mps")[0,:]
wdir =  getvar(ncfile, "wspd_wdir",timeidx=t, units="mps")[1,:]
z = getvar(ncfile, "z")
ter = getvar(ncfile, "ter", timeidx=t)

CLDFRA = getvar(ncfile, "CLDFRA",timeidx=t)
QC = getvar(ncfile, "QCLOUD",timeidx=t)*1000
QI = getvar(ncfile, "QICE",timeidx=t)*1000
RH=wrf.rh(Q/1000, P*100, T, meta=True)
EPT=wrf.eth(Q/1000, T, P*100, meta=True, units='K')
QS=np.divide(Q,RH/100)
EPTS=wrf.eth(QS/1000, T, P*100, meta=True, units='K')


TD = wrf.td(P, Q/1000, meta=True, units='degC')
ht = getvar(ncfile, "z",timeidx=t, units="m")
ht_1000=wrf.vinterp(ncfile,ht,'p',[1000.0, 500.0],field_type='ght',extrapolate=True,timeidx=t)

# Get the latitude and longitude points
lats, lons = latlon_coords(slp)

#TD = wrf.td(pres, q2, meta=True, units='degC')
#RH2=wrf.rh(q2, pres*100, T2, meta=True)
#Thae=wrf.eth(q2, T2, pres*100, meta=True, units='K')
x, y = ll_to_xy(ncfile, 59.85, 17.64)
#if ds == 2:
 #   x=30
  #  y=30
lat_lon = xy_to_ll(ncfile, x, y)
#print(x,y)
#print(lat_lon)
zU=z[:,y,x]

#print(zU)

zp2=np.sqrt(zU)

#zp2=zp2.transpose(transpose_coords=True, missing_dims='raise')

xz=np.arange(1,m+1)
yz=np.arange(1,n+1)
#print(xz)
xZ=np.zeros((39,1),dtype=xz.dtype) + xz
yZ=np.zeros((39,1),dtype=xz.dtype) + yz

#point
TU=xarray.DataArray.to_numpy(T[:,y,x]-273.15)
#print(TU)
TdU=TD[:,y,x]
UU=wspd[:,y,x]
WDU=wdir[:,y,x]
T2U=np.array([xarray.DataArray.to_numpy(T2[y,x])])
Td2U=np.array([xarray.DataArray.to_numpy(TD2[y,x])])
#print(T2U)
zu2=xarray.DataArray.to_numpy(zp2)
z2m=np.array([np.sqrt(2)])
#print(z2m)
zu2=np.concatenate([z2m,zu2])
Tu=np.concatenate([T2U,TU])
Tdu=np.concatenate([Td2U,TdU])
#numpy.concatenate([a,b])
#print(Tu)
#print(zu2)

EPTU=xarray.DataArray.to_numpy(EPT[:,y,x])-273.25
EPTSU=xarray.DataArray.to_numpy(EPTS[:,y,x])-273.15

zasq=range(1,85,1)
za=np.square(zasq)
Ta=np.array([-10-za*9.76/1000,0-za*9.76/1000,10-za*9.76/1000,20-za*9.76/1000,30-za*9.76/1000])
#print(za)
Ta=Ta.transpose()
#print(Ta)

terxz=xarray.DataArray.to_numpy(ter[:,x])
terxz2=np.sqrt(terxz)
print(terxz2.shape)
Zxz=xarray.DataArray.to_numpy(z[:,:,x])
Zxz2=np.sqrt(Zxz)
#T2xz=T2[:,x]
QCxz=QC[:,:,x]
#QCxz=QCxz.transpose(transpose_coords=True, missing_dims='raise')
#print(QCxz.shape)
QIxz=QI[:,:,x]
#QIxz=QIxz.transpose(transpose_coords=True, missing_dims='raise')
RHxz=RH[:,:,x]
#RHxz=RHxz.transpose(transpose_coords=True, missing_dims='raise')
Txz=T[:,:,x]-273.15
#Txz=Txz.transpose(transpose_coords=True, missing_dims='raise')
EPTxz=EPT[:,:,x]-273.15
#EPTxz=EPTxz.transpose(transpose_coords=True, missing_dims='raise')
CLDFRAxz=CLDFRA[:,:,x]
#CLDFRAxz=CLDFRAxz.transpose(transpose_coords=True, missing_dims='raise')
RHixz=np.maximum(RHxz-Txz-0.004*np.square(Txz),RHxz)
#print(zp2.shape,RHixz.shape)

teryz=xarray.DataArray.to_numpy(ter[y,:])
teryz2=np.sqrt(teryz)
print(teryz2.shape)
Zyz=xarray.DataArray.to_numpy(z[:,y,:])
Zyz2=np.sqrt(Zyz)
QCyz=QC[:,y,:]
#QCxz=QCxz.transpose(transpose_coords=True, missing_dims='raise')
#print(QCyz.shape)
QIyz=QI[:,y,:]
#QIxz=QIxz.transpose(transpose_coords=True, missing_dims='raise')
RHyz=RH[:,y,:]
#RHxz=RHxz.transpose(transpose_coords=True, missing_dims='raise')
Tyz=T[:,y,:]-273.15
#Txz=Txz.transpose(transpose_coords=True, missing_dims='raise')
EPTyz=EPT[:,y,:]-273.15
#EPTxz=EPTxz.transpose(transpose_coords=True, missing_dims='raise')
CLDFRAyz=CLDFRA[:,y,:]
#CLDFRAxz=CLDFRAxz.transpose(transpose_coords=True, missing_dims='raise')
RHiyz=np.maximum(RHyz-Tyz-0.004*np.square(Tyz),RHyz)
#print(zp2.shape,RHiyz.shape)

bx=10;

# Create a figure
fig = plt.figure(0) #,figsize=(6,6))
ax=fig.add_subplot(1,1,1)
p=plt.plot(TU, zU,color='red')
p2=plt.plot(TdU, zU)
plt.plot(EPTU, zU,color='pink')
plt.plot(EPTSU, zU,color='green')
#ax.clabel(p, p.levels, inline=True, fontsize=10)
#plt.colorbar(T2_contours, ax=ax)
ax.set_title('Sounding at ' + hh + ":00 SNT " + datestring)
ax.set_xticks(range(-70,51,10))
ax.set_xlim(-70,51)
ax.set_ylim(0,13001)
#ax.set_ylim(0,np.sqrt(13000))
#yticks=[50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
#ax.set_yticks(np.sqrt(yticks))
#ax.set_yticklabels(yticks)
ax.grid(True)
#plt.savefig('sond.png', dpi=100, bbox_inches='tight')

# Create a figure
fig = plt.figure(1) #,figsize=(6,6))
ax=fig.add_subplot(1,1,1)
p=plt.plot(Tu, zu2,color='red')
p2=plt.plot(Tdu, zu2)
plt.plot(EPTU, zp2,color='pink')
plt.plot(EPTSU, zp2,color='magenta')
plt.plot(UU, zp2,color='cyan')
plt.plot(WDU/10, zp2,'o',color='green')
plt.plot(Ta, zasq,linestyle='dashed',color='black')
#ax.clabel(p, p.levels, inline=True, fontsize=10)
#plt.colorbar(T2_contours, ax=ax)
ax.set_title('Sounding at ' + hh + ":00 SNT " + datestring)
ax.set_xlim(-70,70)
ax.set_xticks(range(-70,71,10))
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)
ax.grid(True)
#plt.savefig('sond2.png', dpi=100, bbox_inches='tight')

if do==1:
  file='crosses/sound_'+steps+'.png'
  print(file)
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='crosses/sound_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')



fig = plt.figure(2,figsize=(16,12))
ax=fig.add_subplot(2,2,3)
RH_contours = ax.contourf(xZ,Zxz2,RHixz,levels=range(88,109,2),cmap=get_cmap("YlGn"), extend="max")
plt.colorbar(RH_contours, ax=ax)
ax.contour(xZ,Zxz2,(QCxz),levels=[.003,.01, .03, .1, .3, 1, 3])
CQ=ax.contour(xZ,Zxz2,(QIxz),levels=[.0003,.001, .003, .01, .03, .1, 0.3], colors="cyan")
CS=ax.contour(xZ,Zxz2,(Txz),linestyles='dashed',levels=[-40, -20, -12, -6, 0, 5, 10, 15, 20, 25, 30], colors="black")
ax.plot([y,y],[0,zp2[-1]],'k')
ht_fill = ax.fill_between(xz, 0, to_np(terxz2),facecolor="saddlebrown")
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
#ax.clabel(CQ, CQ.levels, inline=True, fontsize=10)
ax.set_title('RH at ' + hh + ":00 SNT " + datestring)
ax.set_xlabel("y")
#ax.set_xticks(xt)
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)
ax.grid(True)

ax=fig.add_subplot(2,2,4)
EPT_contours = ax.contourf(xZ,Zxz2,EPTxz,levels=range(-15,58,3),cmap=get_cmap("hsv_r"), extend="max")
plt.colorbar(EPT_contours, ax=ax)
CQ=ax.contour(xZ,Zxz2,(CLDFRAxz),levels=[0.05, .25], colors="white", linewidths=.5)
CQ1=ax.contour(xZ,Zxz2,(CLDFRAxz),levels=[0.5, .75], colors="white", linewidths=1)
CQ2=ax.contour(xZ,Zxz2,(CLDFRAxz),levels=[0.95], colors="white", linewidths=2)
ax.contour(xZ,Zxz2,EPTxz,levels=range(0,73,1), colors='black', linewidths=0.25)
ax.contour(xZ,Zxz2,EPTxz,levels=range(0,73,15), colors='black', linewidths=1)
ax.plot([y,y],[0,zp2[-1]],'k')
CS=ax.contour(xZ,Zxz2,(Txz),linestyles='dashed',levels=[-40, -20, -12, -6, 0, 5, 10, 15, 20, 25, 30], colors="black")
ht_fill = ax.fill_between(xz, 0, to_np(terxz2),facecolor="saddlebrown")
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
ax.set_title('EPT at ' + hh + ":00 SNT " + datestring)
ax.set_xlabel("y")
#ax.set_xticks(xt)
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)

ax=fig.add_subplot(2,2,1)
RH_contours = ax.contourf(yZ,Zyz2,RHiyz,levels=range(88,109,2),cmap=get_cmap("YlGn"), extend="max")
plt.colorbar(RH_contours, ax=ax)
ax.contour(yZ,Zyz2,(QCyz),levels=[.003,.01, .03, .1, .3, 1, 3])
CQ=ax.contour(yZ,Zyz2,(QIyz),levels=[.0003,.001, .003, .01, .03, .1, 0.3], colors="cyan")
CS=ax.contour(yZ,Zyz2,(Tyz),linestyles='dashed',levels=[-40, -20, -12, -6, 0, 5, 10, 15, 20, 25, 30], colors="black")
ax.plot([x,x],[0,zp2[-1]],'k')
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
#ax.clabel(CQ, CQ.levels, inline=True, fontsize=10)
ht_fill = ax.fill_between(yz, 0, to_np(teryz2),facecolor="saddlebrown")
ax.set_title('RH at ' + hh + ":00 SNT " + datestring)
ax.set_xlabel("x")
#ax.set_xticks(xt)
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)
ax.grid(True)

ax=fig.add_subplot(2,2,2)
EPT_contours = ax.contourf(yZ,Zyz2,EPTyz,levels=range(-15,58,3),cmap=get_cmap("hsv_r"), extend="max")
plt.colorbar(EPT_contours, ax=ax)
CQ=ax.contour(yZ,Zyz2,(CLDFRAyz),levels=[0.05, .25], colors="white", linewidths=.5)
CQ1=ax.contour(yZ,Zyz2,(CLDFRAyz),levels=[0.5, .75], colors="white", linewidths=1)
CQ2=ax.contour(yZ,Zyz2,(CLDFRAyz),levels=[0.95], colors="white", linewidths=2)
ax.contour(yZ,Zyz2,EPTyz,levels=range(0,73,1), colors='black', linewidths=0.25)
ax.contour(yZ,Zyz2,EPTyz,levels=range(0,73,15), colors='black', linewidths=1)
CS=ax.contour(yZ,Zyz2,(Tyz),linestyles='dashed',levels=[-40, -20, -12, -6, 0, 5, 10, 15, 20, 25, 30], colors="black")
ax.plot([x,x],[0,zp2[-1]],'k')
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
ht_fill = ax.fill_between(yz, 0, to_np(teryz2),facecolor="saddlebrown")
ax.set_title('EPT at ' + hh + ":00 SNT " + datestring)
ax.set_xlabel("x")
#ax.set_xticks(xt)
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)

#plt.savefig('cross.png', dpi=100, bbox_inches='tight')

if do==1:
  file='crosses/cross_'+steps+'.png'
  print(file)
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='crosses/cross_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')

# Create a figure
#fig = plt.figure(3) #,figsize=(6,6))
#ax=fig.add_subplot(1,1,1)

print(zU)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot((WDU[:40]-180)*np.pi/180, UU[:40])
ax.plot((WDU[:26]-180)*np.pi/180, UU[:26],color='cyan')
ax.plot((WDU[:18]-180)*np.pi/180, UU[:18],color='red')
ax.plot((WDU[:11]-180)*np.pi/180, UU[:11],color='green')
ax.set_rmax(30)
ax.set_rticks([10, 20, 30])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
ax.grid(True)

ax.set_title("Vector hodograph at " + hh + ":00 SNT " + datestring, va='bottom')

#plt.savefig('hodo.png', dpi=100, bbox_inches='tight')

if do==1:
  file='crosses/hodo_'+steps+'.png'
  print(file)
  plt.savefig(file, dpi=100, bbox_inches='tight')
else:
  file='crosses/hodo_2_'+steps+'.png'
  plt.savefig(file, dpi=100, bbox_inches='tight')


pid=os.fork()
if pid==0:
  plt.show()





