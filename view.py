# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:04:07 2022

@author: opio
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

file1 = np.load('C:/python_work/phd/paper3/real_work/so2_output/so2_obs/obs_2009_12_dec.npy')
file2 = np.load('C:/python_work/phd/paper3/real_work/so2_output/so2_obs/obs_2015_1_jan.npy')
file3 = np.load('C:/python_work/phd/paper3/real_work/so2_output/so2_obs/obs_2010_2_feb.npy')

final = (file1 + file2 + file3)/3

coord = xr.Dataset({'lat': (['lat'], np.arange(-7, 1.2, 0.086)),
                     'lon': (['lon'], np.arange(23.4, 31.6, 0.086))})

fig=plt.figure(figsize=(8, 8))#, dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree())
plt.pcolormesh(coord['lon'], coord['lat'], file2, cmap='gist_ncar_r', vmin=0, vmax=4.5)
ax.coastlines(resolution='10m', color='black', linewidth=0.9)
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
cbar = plt.colorbar(ax=ax, shrink=.62)
cbar.set_label('DU')
plt.xlim([23.4, 31.6])
plt.ylim([-7, 1.2])
plt.show()