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
#import coltbls as coltbls 
import wrf
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords, ALL_TIMES , xy_to_ll, ll_to_xy) 
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.axes as maxes
import xarray 
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cfeature

import convpar


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
    
    

#dos='1'
#ms='09'
#ds='08'
#hs='00'
ms,ds, hs, dos = input("").split()
mm=int(ms)
dd=int(ds)
do=int(dos)
h0=int(hs)
ncfile=netcdf.netcdf_file("wrfout_d0" + dos + "_2023-" + ms + "-" + ds + "_" + hs + ":00:00s",'r')
#ncfile=netcdf.netcdf_file("out2.nc",'r')

time = np.round(6*(getvar(ncfile, "XTIME",ALL_TIMES)/60+h0+1))/6
#print(time)
dat0=datetime(2023,mm,dd,h0,0)
i=0
#for tim in time:
#	dat[i,:]=dat0+timedelta(0,tim*3600)
#	i+=1
print(dat0)
print(h0)
#if h0==0:
#    time=time.where(time<=24, time-24)
if h0==18:
    time=time.where(time<24, time-24)

print(time)
T2 = getvar(ncfile, "T2",ALL_TIMES)
q2 = getvar(ncfile, "Q2",ALL_TIMES)
u10 = getvar(ncfile, "U10",ALL_TIMES)
v10 = getvar(ncfile, "V10",ALL_TIMES)
U10 = np.sqrt(u10**2 +v10**2)
#u10E = getvar(ncfile, "U10E",ALL_TIMES)
#v10E = getvar(ncfile, "V10E",ALL_TIMES)
#U10E = np.sqrt(u10E**2 +v10E**2)
WD10 = getvar(ncfile, "uvmet10_wdir",ALL_TIMES)
pres = getvar(ncfile, "slp",ALL_TIMES)
TSK = getvar(ncfile, "TSK",ALL_TIMES)

SW = getvar(ncfile, "SWDOWN",ALL_TIMES)
LW = getvar(ncfile, "GLW",ALL_TIMES)
ALB = getvar(ncfile, "ALBEDO",ALL_TIMES)
EMIS = getvar(ncfile, "EMISS",ALL_TIMES)
PBLH = getvar(ncfile, "PBLH",ALL_TIMES)
HFX = getvar(ncfile, "HFX",ALL_TIMES)
LHFX = getvar(ncfile, "LH",ALL_TIMES)

rains = getvar(ncfile, "RAINNC",ALL_TIMES)
rainc = getvar(ncfile, "RAINC",ALL_TIMES)
rain=rains+rainc

#3d fields
CLDFRA = getvar(ncfile, "CLDFRA",ALL_TIMES)
QC = getvar(ncfile, "QCLOUD",ALL_TIMES)*1000
z = getvar(ncfile, "z")
QI = getvar(ncfile, "QICE",ALL_TIMES)*1000
Q = getvar(ncfile, "QVAPOR",ALL_TIMES)*1000
QS = getvar(ncfile, "QSNOW",ALL_TIMES)*1000
QR = getvar(ncfile, "QRAIN",ALL_TIMES)*1000
QG = getvar(ncfile, "QGRAUP",ALL_TIMES)*1000
TH = getvar(ncfile, "th",ALL_TIMES)
P = getvar(ncfile, "pressure",ALL_TIMES)
T = getvar(ncfile,"tk",ALL_TIMES) 
RH=wrf.rh(Q/1000, P*100, T, meta=True)
EPT=wrf.eth(Q/1000, T, P*100, meta=True, units='K')
PT=getvar(ncfile, "th",ALL_TIMES, units='degC')

print(RH.shape)
ncfile.close

slp=pres[1,:,:]
# Smooth the sea level pressure since it tends to be noisy near the
# mountains
smooth_slp = smooth2d(slp, 3, cenweight=4)

# Get the latitude and longitude points
lats, lons = latlon_coords(slp)

qdiff=.0e-3

TD = wrf.td(pres, q2+qdiff, meta=True, units='degC')
RH2=wrf.rh(q2+qdiff, pres*100, T2, meta=True)
Thae=wrf.eth(q2+qdiff, T2, pres*100, meta=True, units='K')
x, y = ll_to_xy(ncfile, 59.85, 17.64)
#x, y = ll_to_xy(ncfile, 60.58, 15.67) #Falun
#if ds == 2:
 #   x=30
  #  y=30
lat_lon = xy_to_ll(ncfile, x, y)
print(x,y)
print(lat_lon)

#T2n=np.array(T2)
T2U=T2[:,y,x]-273.15
TdU=TD[:,y,x]
ThaeU=Thae[:,y,x]-273.15
U10U=U10[:,y,x]
#U10EU=U10E[:,y,x]
WD10U=WD10[:,y,x]
rainU=np.diff(rain[:,y,x])

RH2U=RH2[:,y,x]
TSKU=TSK[:,y,x]-273.15
SWU=SW[:,y,x]
LWU=LW[:,y,x]
ALBU=ALB[:,y,x]
EMISU=EMIS[:,y,x]
PBLHU=PBLH[:,y,x]

#rainp=squeeze(rain(xp,yp,2:xt)-rain(xp,yp,1:xt-1));

#[Td,Thae,q]=Td_ekvT(T2,r,p)
#print(T2U)
QCtz=QC[:,:,y,x]
QCtz=QCtz.transpose(transpose_coords=True, missing_dims='raise')
print(QCtz.shape)
QItz=QI[:,:,y,x]
QItz=QItz.transpose(transpose_coords=True, missing_dims='raise')
QRtz=QR[:,:,y,x]
QRtz=QRtz.transpose(transpose_coords=True, missing_dims='raise')
QStz=QS[:,:,y,x]
QStz=QStz.transpose(transpose_coords=True, missing_dims='raise')
QGtz=QG[:,:,y,x]
QGtz=QGtz.transpose(transpose_coords=True, missing_dims='raise')
CLDFRAtz=CLDFRA[:,:,y,x]
CLDFRAtz=CLDFRAtz.transpose(transpose_coords=True, missing_dims='raise')
RHtz=RH[:,:,y,x]
RHtz=RHtz.transpose(transpose_coords=True, missing_dims='raise')
Ttz=T[:,:,y,x]-273.15
Ttz=Ttz.transpose(transpose_coords=True, missing_dims='raise')
PTtz=PT[:,:,y,x]
PTtz=PTtz.transpose(transpose_coords=True, missing_dims='raise')
EPTtz=EPT[:,:,y,x]-273.15
EPTtz=EPTtz.transpose(transpose_coords=True, missing_dims='raise')
del EPT, QG, QS, QR, QI, QC, CLDFRA, TH, 
Qs=np.divide(Q,RH/100)
EPTS=wrf.eth(Qs/1000, T, P*100, meta=True, units='K')
EPTStz=EPTS[:,:,y,x]-273.15
EPTStz=EPTStz.transpose(transpose_coords=True, missing_dims='raise')
EPTS=wrf.eth(Qs/1000, T, P*100, meta=True, units='K')
EPTStz=EPTS[:,:,y,x]-273.15
EPTStz=EPTStz.transpose(transpose_coords=True, missing_dims='raise')


RHitz=np.maximum(RHtz-Ttz-0.004*np.square(Ttz),RHtz)
zp=z[:,y,x]

zp2=np.sqrt(zp)

EPTStz=xarray.DataArray.to_numpy(EPTStz)
ThaeU=xarray.DataArray.to_numpy(ThaeU)

#print(EPTStz)
#print(ThaeU.shape)
LFC=np.zeros((len(time)))
LFC2=np.zeros((len(time)))
LEQ=np.zeros((len(time)))
LEQ2=np.zeros((len(time)))
TEQ=np.zeros((len(time)))
for i in np.arange(len(time)):
  print(i)
  #print(ThaeU[i])
  #print(EPTStz[:,i])
  #print(zp)
  #print(zp2)
  #print(Ttz[:,i])
  LFC[i], LEQ[i], TEQ[i], LFC2[i], LEQ2[i] = convpar.convpar(ThaeU[i],EPTStz[:,i],zp,zp2,Ttz[:,i])

print(LFC)
print(LEQ)
print(TEQ)


if dos=='1':
    xt=range(12,51,2)
elif h0<18:
    xt=range(h0+6,h0+25,2)
else:
    xt=range(0,19,2)

fig=plt.figure(1,figsize=(5,12))
sb=7
#plt.subplots(sb,1, sharex='col')
ax=fig.add_subplot(sb,1,1)
ax.plot(time,TdU, label='Td') 
ax.plot(time,T2U, label='T2') 
ax.plot(time,ThaeU, label='\u03B8_ae')
#ax.plot(time,TSKU, label='Tsk') 
ax.plot(time,T2U*0,color='black')
ax.legend(loc='best')
ax.set_title('Uppsala WRF')
ax.set_ylabel("Temp")
ax.set_xticks(xt)
#ax.set_xticklabel("")
ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True)

ax=fig.add_subplot(sb,1,2)
ax.plot(time,RH2U, label='RH')
ax.plot(time,100+TEQ, label='100+TEQ')
ax.plot(time,T2U*0,color='black')
ax.legend(loc='best')
ax.set_ylabel("RH")
ax.set_ylim(30,100)
ax.set_yticks(range(40,101,10))
ax.set_xticks(xt)
#ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
#ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True)

ax=fig.add_subplot(sb,1,3)
ax.plot(time,U10U, label='WS') 
#ax.plot(time,U10EU, label='WS')
ax.set_ylabel("WS")
ax.set_ylim(0,16)
ax.set_yticks(range(0,17,2))
ax.set_xticks(xt)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True)

ax=fig.add_subplot(sb,1,4)
ax.plot(time,WD10U, 'o', label='WD') 
ax.set_ylabel("WD")
ax.set_ylim(0,360)
ax.set_yticks(range(0,361,90))
ax.set_xticks(xt)
ax.grid(True)

ax=fig.add_subplot(sb,1,5)
ax.plot(time,SWU, label='SW') 
ax.plot(time,LWU, label='LW') 
ax.legend(loc='best')
ax.set_ylabel("Rad")
ax.set_ylim(0,400)
ax.set_yticks(range(0,401,100))
ax.set_xticks(xt)
ax.grid(True)

ax=fig.add_subplot(sb,1,6)
ax.plot(time,130*(T2U-TdU),label='LCL') 
ax.plot(time,PBLHU,label='BLH')
ax.plot(time,LFC, label='LFC')
ax.plot(time,LEQ, label='LEQ')
ax.legend(loc='best')
ax.set_ylabel("LCL/BLH")
ax.set_ylim(0,1800)
ax.set_yticks(range(0,1801,300))
ax.set_xticks(xt)
ax.grid(True)

ax=fig.add_subplot(sb,1,7)
ax.plot(time[1:],np.sqrt(rainU) )
ax.set_ylabel("precip")
#ax.set_ylim(0,4)
#ax.set_yticks(range(0,4,1))
yticks=[0, .1, .2, .5,  1., 2., 5., 10]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)
ax.set_xticks(xt)
ax.grid(True)

plt.savefig('wrf.png', dpi=100, bbox_inches='tight')




fig=plt.figure(2,figsize=(5,5))
ax=fig.add_subplot(1,1,1)
RH_contours = ax.contourf(time,zp2,RHitz,levels=range(88,109,2),cmap=get_cmap("YlGn"), extend="max")
plt.colorbar(RH_contours, ax=ax)
ax.contour(time,zp2,(QCtz),levels=[.003,.01, .03, .1, .3, 1, 3])
CQ=ax.contour(time,zp2,(QItz),levels=[.0003,.001, .003, .01, .03, .1, 0.3], colors="cyan")
CS=ax.contour(time,zp2,(QStz),levels=[.0003,.001, .003, .01, .03, .1, 0.3], colors="white")
CS=ax.contour(time,zp2,(Ttz),linestyles='dashed',levels=[-40, -20, -12, -6, 0, 5, 10, 15, 20, 25, 30], colors="black")
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
#ax.clabel(CQ, CQ.levels, inline=True, fontsize=10)
ax.plot(time,np.sqrt(130*(T2U-TdU)),label='LCL') 
ax.plot(time,np.sqrt(PBLHU),label='BLH')
ax.plot(time,LFC2, label='LFC')
ax.plot(time,LEQ2, label='LEQ')
ax.legend(loc='best')
ax.set_xticks(xt)
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)
ax.grid(True)
plt.savefig('wrf2.png', dpi=100, bbox_inches='tight')


print(time.shape,zp2.shape,EPTtz.shape)

fig=plt.figure(3,figsize=(5,5))
ax=fig.add_subplot(1,1,1)
EPT_contours = ax.contourf(time,zp2,EPTtz,levels=range(-15,58,3),cmap=get_cmap("hsv_r"), extend="max")
#cmap=get_cmap("hsv_r")
plt.colorbar(EPT_contours, ax=ax)
CQ=ax.contour(time,zp2,(CLDFRAtz),levels=[0.05, .25], colors="white", linewidths=.5)
CQ1=ax.contour(time,zp2,(CLDFRAtz),levels=[0.5, .75], colors="white", linewidths=1)
CQ2=ax.contour(time,zp2,(CLDFRAtz),levels=[0.95], colors="white", linewidths=2)
ax.contour(time,zp2,EPTtz,levels=range(0,73,1), colors='black', linewidths=0.25)
ax.contour(time,zp2,EPTtz,levels=range(0,73,15), colors='black', linewidths=1)
CS=ax.contour(time,zp2,(Ttz),linestyles='dashed',levels=[-40, -20, -12, -6, 0, 5, 10, 15, 20, 25, 30], colors="black")
ax.plot(time,np.sqrt(130*(T2U-TdU)),label='LCL') 
ax.plot(time,np.sqrt(PBLHU),label='BLH')
ax.plot(time,LFC2, label='LFC')
ax.plot(time,LEQ2, label='LEQ')
ax.legend(loc='best')
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
ax.set_xticks(xt)
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)

ax.grid(True)
plt.savefig('wrf3.png', dpi=100, bbox_inches='tight')

fig=plt.figure(4,figsize=(5,12))
ax=fig.add_subplot(2,1,1)
EPT_contours = ax.contourf(time,zp2,PTtz,levels=range(-24,37,3),cmap=get_cmap("hsv_r"), extend="both")
plt.colorbar(EPT_contours, ax=ax)
CQ=ax.contour(time,zp2,(CLDFRAtz),levels=[0.05, .25], colors="white", linewidths=.5)
CQ1=ax.contour(time,zp2,(CLDFRAtz),levels=[0.5, .75], colors="white", linewidths=1)
CQ2=ax.contour(time,zp2,(CLDFRAtz),levels=[0.95], colors="white", linewidths=2)
ax.contour(time,zp2,PTtz,levels=range(0,73,1), colors='black', linewidths=0.25)
ax.contour(time,zp2,PTtz,levels=range(0,73,15), colors='black', linewidths=1)
CS=ax.contour(time,zp2,(Ttz),linestyles='dashed',levels=[-40, -20, -12, -6, 0, 5, 10, 15, 20, 25, 30], colors="black")
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
ax.plot(time,np.sqrt(130*(T2U-TdU)),label='LCL') 
ax.plot(time,np.sqrt(PBLHU),label='BLH')
ax.legend(loc='best')
ax.set_xticks(xt)
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)
ax.grid(True)

ax=fig.add_subplot(2,1,2)
EPT_contours = ax.contourf(time,zp2,EPTStz,levels=range(-15,58,3),cmap=get_cmap("hsv_r"), extend="max")
plt.colorbar(EPT_contours, ax=ax)
CQ=ax.contour(time,zp2,(CLDFRAtz),levels=[0.05, .25], colors="white", linewidths=.5)
CQ1=ax.contour(time,zp2,(CLDFRAtz),levels=[0.5, .75], colors="white", linewidths=1)
CQ2=ax.contour(time,zp2,(CLDFRAtz),levels=[0.95], colors="white", linewidths=2)
ax.contour(time,zp2,EPTStz,levels=range(0,73,1), colors='black', linewidths=0.25)
ax.contour(time,zp2,EPTStz,levels=range(0,73,15), colors='black', linewidths=1)
CS=ax.contour(time,zp2,(Ttz),linestyles='dashed',levels=[-40, -20, -12, -6, 0, 5, 10, 15, 20, 25, 30], colors="black")
ax.plot(time,np.sqrt(130*(T2U-TdU)),label='LCL') 
ax.plot(time,np.sqrt(PBLHU),label='BLH')
ax.plot(time,LFC2, label='LFC')
ax.plot(time,LEQ2, label='LEQ')
ax.legend(loc='best')
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
ax.set_xticks(xt)
#ax.set_ylim(0,13001)
ax.set_ylim(0,np.sqrt(13000))
yticks=[50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000]
ax.set_yticks(np.sqrt(yticks))
ax.set_yticklabels(yticks)
ax.grid(True)

plt.savefig('wrf4.png', dpi=100, bbox_inches='tight')

plt.show()

