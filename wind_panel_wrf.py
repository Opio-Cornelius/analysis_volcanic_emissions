#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 23:51:27 2022

@author: oronald
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature

from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,
                 cartopy_xlim, cartopy_ylim)

# Open the NetCDF file
ncfile = Dataset('/home/oronald/model/so2_results/wrf_2014_9_sep.nc')
ncfile2 = Dataset('/home/oronald/model/so2_results/wrf_2014_10_oct.nc')
ncfile3 = Dataset('/home/oronald/model/so2_results/wrf_2014_11_nov.nc')

ncfile4 = Dataset('/home/oronald/model/so2_results/wrf_2014_12_dec.nc')
ncfile5 = Dataset('/home/oronald/model/so2_results/wrf_2015_1_jan.nc')
ncfile6 = Dataset('/home/oronald/model/so2_results/wrf_2015_2_feb.nc')

ncfile7 = Dataset('/home/oronald/model/so2_results/wrf_2015_3_mar.nc')
ncfile8 = Dataset('/home/oronald/model/so2_results/wrf_2015_4_apr.nc')
ncfile9 = Dataset('/home/oronald/model/so2_results/wrf_2015_5_may.nc')

ncfile10 = Dataset('/home/oronald/model/so2_results/wrf_2015_6_june.nc')
ncfile11 = Dataset('/home/oronald/model/so2_results/wrf_2015_7_july.nc')
ncfile12 = Dataset('/home/oronald/model/so2_results/wrf_2015_8_aug.nc')



# Extract the pressure, geopotential height, and wind variables
p = getvar(ncfile, "pressure")
z = getvar(ncfile, "z", units="dm")
ua = getvar(ncfile, "ua", units="kt")
va = getvar(ncfile, "va", units="kt")
wspd = (getvar(ncfile, "wspd_wdir", units="kts")[0,:])*0.51
ht_600 = interplevel(z, p, 600)
u_600 = interplevel(ua, p, 600)
v_600 = interplevel(va, p, 600)
wspd_600 = interplevel(wspd, p, 600)
lats, lons = latlon_coords(ht_600)
cart_proj = get_cartopy(ht_600)


p2 = getvar(ncfile2, "pressure")
z2 = getvar(ncfile2, "z", units="dm")
ua2 = getvar(ncfile2, "ua", units="kt")
va2 = getvar(ncfile2, "va", units="kt")
wspd2 = (getvar(ncfile2, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_2 = interplevel(z2, p2, 600)
u_600_2 = interplevel(ua2, p2, 600)
v_600_2 = interplevel(va2, p2, 600)
wspd_600_2 = interplevel(wspd2, p2, 600)
lats_2, lons_2 = latlon_coords(ht_600_2)
cart_proj_2 = get_cartopy(ht_600_2)


p3 = getvar(ncfile3, "pressure")
z3 = getvar(ncfile3, "z", units="dm")
ua3 = getvar(ncfile3, "ua", units="kt")
va3 = getvar(ncfile3, "va", units="kt")
wspd3 = (getvar(ncfile3, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_3 = interplevel(z3, p3, 600)
u_600_3 = interplevel(ua3, p3, 600)
v_600_3 = interplevel(va3, p3, 600)
wspd_600_3 = interplevel(wspd3, p3, 600)
lats_3, lons_3 = latlon_coords(ht_600_3)
cart_proj_3 = get_cartopy(ht_600_3)


p4 = getvar(ncfile4, "pressure")
z4 = getvar(ncfile4, "z", units="dm")
ua4 = getvar(ncfile4, "ua", units="kt")
va4 = getvar(ncfile4, "va", units="kt")
wspd4 = (getvar(ncfile4, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_4 = interplevel(z4, p4, 600)
u_600_4 = interplevel(ua4, p4, 600)
v_600_4 = interplevel(va4, p4, 600)
wspd_600_4 = interplevel(wspd4, p4, 600)
lats_4, lons_4 = latlon_coords(ht_600_4)
cart_proj_4 = get_cartopy(ht_600_4)


p5 = getvar(ncfile5, "pressure")
z5 = getvar(ncfile5, "z", units="dm")
ua5 = getvar(ncfile5, "ua", units="kt")
va5 = getvar(ncfile5, "va", units="kt")
wspd5 = (getvar(ncfile5, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_5 = interplevel(z5, p5, 600)
u_600_5 = interplevel(ua5, p5, 600)
v_600_5 = interplevel(va5, p5, 600)
wspd_600_5 = interplevel(wspd5, p5, 600)
lats_5, lons_5 = latlon_coords(ht_600_5)
cart_proj_5 = get_cartopy(ht_600_5)


p6 = getvar(ncfile6, "pressure")
z6 = getvar(ncfile6, "z", units="dm")
ua6 = getvar(ncfile6, "ua", units="kt")
va6 = getvar(ncfile6, "va", units="kt")
wspd6 = (getvar(ncfile6, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_6 = interplevel(z6, p6, 600)
u_600_6 = interplevel(ua6, p6, 600)
v_600_6 = interplevel(va6, p6, 600)
wspd_600_6 = interplevel(wspd6, p6, 600)
lats_6, lons_6 = latlon_coords(ht_600_6)
cart_proj_6 = get_cartopy(ht_600_6)


p7 = getvar(ncfile7, "pressure")
z7 = getvar(ncfile7, "z", units="dm")
ua7 = getvar(ncfile7, "ua", units="kt")
va7 = getvar(ncfile7, "va", units="kt")
wspd7 = (getvar(ncfile7, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_7 = interplevel(z7, p7, 600)
u_600_7 = interplevel(ua7, p7, 600)
v_600_7 = interplevel(va7, p7, 600)
wspd_600_7 = interplevel(wspd7, p7, 600)
lats_7, lons_7 = latlon_coords(ht_600_7)
cart_proj_7 = get_cartopy(ht_600_7)


p8 = getvar(ncfile8, "pressure")
z8 = getvar(ncfile8, "z", units="dm")
ua8 = getvar(ncfile8, "ua", units="kt")
va8 = getvar(ncfile8, "va", units="kt")
wspd8 = (getvar(ncfile8, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_8 = interplevel(z8, p8, 600)
u_600_8 = interplevel(ua8, p8, 600)
v_600_8 = interplevel(va8, p8, 600)
wspd_600_8 = interplevel(wspd8, p8, 600)
lats_8, lons_8 = latlon_coords(ht_600_8)
cart_proj_8 = get_cartopy(ht_600_8)


p9 = getvar(ncfile9, "pressure")
z9 = getvar(ncfile9, "z", units="dm")
ua9 = getvar(ncfile9, "ua", units="kt")
va9 = getvar(ncfile9, "va", units="kt")
wspd9 = (getvar(ncfile9, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_9 = interplevel(z9, p9, 600)
u_600_9 = interplevel(ua9, p9, 600)
v_600_9 = interplevel(va9, p9, 600)
wspd_600_9 = interplevel(wspd9, p9, 600)
lats_9, lons_9 = latlon_coords(ht_600_9)
cart_proj_9 = get_cartopy(ht_600_9)

p10 = getvar(ncfile10, "pressure")
z10 = getvar(ncfile10, "z", units="dm")
ua10 = getvar(ncfile10, "ua", units="kt")
va10 = getvar(ncfile10, "va", units="kt")
wspd10 = (getvar(ncfile10, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_10 = interplevel(z10, p10, 600)
u_600_10 = interplevel(ua10, p10, 600)
v_600_10 = interplevel(va10, p10, 600)
wspd_600_10 = interplevel(wspd10, p10, 600)
lats_10, lons_10 = latlon_coords(ht_600_10)
cart_proj_10 = get_cartopy(ht_600_10)


p11 = getvar(ncfile11, "pressure")
z11 = getvar(ncfile11, "z", units="dm")
ua11 = getvar(ncfile11, "ua", units="kt")
va11 = getvar(ncfile11, "va", units="kt")
wspd11 = (getvar(ncfile11, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_11 = interplevel(z11, p11, 600)
u_600_11 = interplevel(ua11, p11, 600)
v_600_11 = interplevel(va11, p11, 600)
wspd_600_11 = interplevel(wspd11, p11, 600)
lats_11, lons_11 = latlon_coords(ht_600_11)
cart_proj_11 = get_cartopy(ht_600_11)


p12 = getvar(ncfile12, "pressure")
z12 = getvar(ncfile12, "z", units="dm")
ua12 = getvar(ncfile12, "ua", units="kt")
va12 = getvar(ncfile12, "va", units="kt")
wspd12 = (getvar(ncfile12, "wspd_wdir", units="kts")[0,:])*0.51
ht_600_12 = interplevel(z12, p12, 600)
u_600_12 = interplevel(ua12, p12, 600)
v_600_12 = interplevel(va12, p12, 600)
wspd_600_12 = interplevel(wspd12, p12, 600)
lats_12, lons_12 = latlon_coords(ht_600_12)
cart_proj_12 = get_cartopy(ht_600_12)





# Calculate averages for each season

#For SON
ht_600_son = (ht_600 + ht_600_2 + ht_600_3)/3
u_600_son = (u_600 + u_600_2 + u_600_3)/3
v_600_son = (v_600 + v_600_2 + v_600_3)/3
wspd_600_son = (wspd_600 + wspd_600_2 + wspd_600_3)/3

#For DJF
ht_600_djf = (ht_600_4 + ht_600_5 + ht_600_6)/3
u_600_djf = (u_600_4 + u_600_5 + u_600_6)/3
v_600_djf = (v_600_4 + v_600_5 + v_600_6)/3
wspd_600_djf = (wspd_600_4 + wspd_600_5 + wspd_600_6)/3

#For MAM
ht_600_mam = (ht_600_7 + ht_600_8 + ht_600_9)/3
u_600_mam = (u_600_7 + u_600_8 + u_600_9)/3
v_600_mam = (v_600_7 + v_600_8 + v_600_9)/3
wspd_600_mam = (wspd_600_7 + wspd_600_8 + wspd_600_9)/3

ht_600_jja = (ht_600_10 + ht_600_11 + ht_600_12)/3
u_600_jja = (u_600_10 + u_600_11 + u_600_12)/3
v_600_jja = (v_600_10 + v_600_11 + v_600_12)/3
wspd_600_jja = (wspd_600_10 + wspd_600_11 + wspd_600_12)/3




# Create the figure
fig = plt.figure(figsize=(16, 4), dpi=500)
#mpl.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0, wspace=0.08)


ax = plt.subplot(1,4,1, projection=cart_proj)
levels = [0, 2, 4, 6, 8, 10, 12, 14]
plt.contourf(to_np(lons), to_np(lats), to_np(wspd_600), cmap='jet',
                             transform=crs.PlateCarree(), levels=levels)
plt.quiver(to_np(lons[::10,::10]), to_np(lats[::10,::10]),
          to_np(u_600_son[::10, ::10]), to_np(v_600_son[::10, ::10]),
          transform=crs.PlateCarree())
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=0.9)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
ax.set_xlim(cartopy_xlim(ht_600))
ax.set_ylim(cartopy_ylim(ht_600))
plt.title('SON')


ax = plt.subplot(1,4,2, projection=cart_proj_4)
plt.contourf(to_np(lons_4), to_np(lats_4), to_np(wspd_600_djf), cmap=get_cmap("jet"),
                             transform=crs.PlateCarree(), levels=levels)
plt.quiver(to_np(lons_4[::10,::10]), to_np(lats_4[::10,::10]),
          to_np(u_600_djf[::10, ::10]), to_np(v_600_djf[::10, ::10]),
          transform=crs.PlateCarree())
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=0.9)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
ax.set_xlim(cartopy_xlim(ht_600_4))
ax.set_ylim(cartopy_ylim(ht_600_4))
plt.title('DJF')


ax = plt.subplot(1,4,3, projection=cart_proj_7)
plt.contourf(to_np(lons_7), to_np(lats_7), to_np(wspd_600_mam), cmap=get_cmap("jet"),
                             transform=crs.PlateCarree(), levels=levels)
plt.quiver(to_np(lons_7[::10,::10]), to_np(lats_7[::10,::10]),
          to_np(u_600_mam[::10, ::10]), to_np(v_600_mam[::10, ::10]),
          transform=crs.PlateCarree())
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=0.9)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
ax.set_xlim(cartopy_xlim(ht_600_7))
ax.set_ylim(cartopy_ylim(ht_600_7))
plt.title('MAM')


ax = plt.subplot(1,4,4, projection=cart_proj_10)
plot_jja = plt.contourf(to_np(lons_10), to_np(lats_10), to_np(wspd_600_jja), cmap=get_cmap("jet"),
                             transform=crs.PlateCarree(), levels=levels)
plt.quiver(to_np(lons_10[::10,::10]), to_np(lats_10[::10,::10]),
          to_np(u_600_jja[::10, ::10]), to_np(v_600_jja[::10, ::10]),
          transform=crs.PlateCarree())
lakes_10m = cfeature.NaturalEarthFeature('physical','lakes','10m')
ax.add_feature(cfeature.BORDERS, linewidth=0.9)
ax.add_feature(lakes_10m, facecolor='none', edgecolor='k')
ax.set_xlim(cartopy_xlim(ht_600_10))
ax.set_ylim(cartopy_ylim(ht_600_10))
plt.title('JJA')
colorbar_axes = plt.gcf().add_axes([0.91, 0.14, 0.011, 0.73])
plt.colorbar(plot_jja, colorbar_axes, orientation='vertical', label='m/s')
plt.show()

# Set the map bounds
#ax.set_xlim(cartopy_xlim(ht_600))
#ax.set_ylim(cartopy_ylim(ht_600))

