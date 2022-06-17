# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 22:39:32 2022

@author: user
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

#Read-in the files
ds1 = xr.open_dataset('C:/python_work/phd/paper3/real_work/wind/file1.nc')
ds2 = xr.open_dataset('C:/python_work/phd/paper3/real_work/wind/file2.nc')
ds3 = xr.open_dataset('C:/python_work/phd/paper3/real_work/wind/file3.nc')
ds4 = xr.open_dataset('C:/python_work/phd/paper3/real_work/wind/file4.nc')
ds5 = xr.open_dataset('C:/python_work/phd/paper3/real_work/wind/file5.nc')


#Make an average of the wind variables

#For SON
wspd_son = np.mean(np.sqrt(np.add(np.square(ds1.u),np.square(ds1.v))),axis=0)
u_son = np.mean(ds1.u,axis=0)
v_son = np.mean(ds1.v,axis=0)

#For DJF
wspd_dec = np.mean(np.sqrt(np.add(np.square(ds2.u),np.square(ds2.v))),axis=0)
u_dec = np.mean(ds2.u,axis=0)
v_dec = np.mean(ds2.v,axis=0)

wspd_jan_feb = np.mean(np.sqrt(np.add(np.square(ds3.u),np.square(ds3.v))),axis=0)
u_jan_feb = np.mean(ds3.u,axis=0)
v_jan_feb = np.mean(ds3.v,axis=0)

wspd_djf = (wspd_dec + wspd_jan_feb)/2
u_djf = (u_dec + u_jan_feb)/2
v_djf = (v_dec + v_jan_feb)/2


#For MAM
wspd_mam = np.mean(np.sqrt(np.add(np.square(ds4.u),np.square(ds4.v))),axis=0)
u_mam = np.mean(ds4.u,axis=0)
v_mam = np.mean(ds4.v,axis=0)


#For JJA
wspd_jja = np.mean(np.sqrt(np.add(np.square(ds5.u),np.square(ds5.v))),axis=0)
u_jja = np.mean(ds5.u,axis=0)
v_jja = np.mean(ds5.v,axis=0)



#setting up the quiver arguments (these display the wind vectors on the plot)
xx_son = wspd_son.longitude.values
yy_son = wspd_son.latitude.values
X_son,Y_son =np.meshgrid(xx_son, yy_son)
U_son = u_son.data
V_son = v_son.data

xx_djf = wspd_djf.longitude.values
yy_djf = wspd_djf.latitude.values
X_djf,Y_djf =np.meshgrid(xx_djf, yy_djf)
U_djf = u_djf.data
V_djf = v_djf.data

xx_mam = wspd_mam.longitude.values
yy_mam = wspd_mam.latitude.values
X_mam,Y_mam =np.meshgrid(xx_mam, yy_mam)
U_mam = u_mam.data
V_mam = v_mam.data


xx_jja = wspd_jja.longitude.values
yy_jja = wspd_jja.latitude.values
X_jja,Y_jja =np.meshgrid(xx_jja, yy_jja)
U_jja = u_jja.data
V_jja = v_jja.data



#Plot out the figures
fig=plt.figure(figsize=(16, 4), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0, wspace=0.08)


ax = plt.subplot(1,4,1, projection=ccrs.PlateCarree())
plt.pcolormesh(wspd_son['longitude'], wspd_son['latitude'], wspd_son, cmap='jet',
               vmin=0, vmax=15)
plt.quiver(X_son[::4, ::4], Y_son[::4, ::4], U_son[::4, ::4], V_son[::4, ::4], 
           transform=ccrs.PlateCarree(), color='k', scale=50, width=0.007)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=0.9)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([23.4, 31.6])
plt.ylim([-7, 1.2])
plt.title('SON')
plt.text(0.107, 0.5, 'ERA5', rotation='vertical',
         transform=plt.gcf().transFigure)

ax = plt.subplot(1,4,2, projection=ccrs.PlateCarree())
plt.pcolormesh(wspd_djf['longitude'], wspd_djf['latitude'], wspd_djf, cmap='jet',
               vmin=0, vmax=15)
plt.quiver(X_djf[::4, ::4], Y_djf[::4, ::4], U_djf[::4, ::4], V_djf[::4, ::4], 
           transform=ccrs.PlateCarree(), color='k', scale=50, width=0.007)
ax.add_feature(cfeature.BORDERS, linewidth=0.9)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([23.4, 31.6])
plt.ylim([-7, 1.2])
plt.title('DJF')

ax = plt.subplot(1,4,3, projection=ccrs.PlateCarree())
plt.pcolormesh(wspd_mam['longitude'], wspd_mam['latitude'], wspd_mam, cmap='jet',
               vmin=0, vmax=15)
plt.quiver(X_mam[::4, ::4], Y_mam[::4, ::4], U_mam[::4, ::4], V_mam[::4, ::4], 
           transform=ccrs.PlateCarree(), color='k', scale=50, width=0.007)
ax.add_feature(cfeature.BORDERS, linewidth=0.9)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([23.4, 31.6])
plt.ylim([-7, 1.2])
plt.title('MAM')

ax = plt.subplot(1,4,4, projection=ccrs.PlateCarree())
plot_jja = plt.pcolormesh(wspd_jja['longitude'], wspd_jja['latitude'], wspd_jja, cmap='jet',
                          vmin=0, vmax=15)
plt.quiver(X_jja[::4, ::4], Y_jja[::4, ::4], U_jja[::4, ::4], V_jja[::4, ::4], 
           transform=ccrs.PlateCarree(), color='k', scale=50, width=0.007)
ax.add_feature(cfeature.BORDERS, linewidth=0.9)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
plt.xlim([23.4, 31.6])
plt.ylim([-7, 1.2])
plt.title('JJA')
colorbar_axes = plt.gcf().add_axes([0.91, 0.14, 0.011, 0.73])
plt.colorbar(plot_jja, colorbar_axes, orientation='vertical', label='m/s')