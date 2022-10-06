# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:25:00 2022

@author: opio
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#OBS
file_son = xr.open_dataset('C:/python_work/phd/paper3/phase3_approved/cross_section_phase3/son.nc')
data_obs_son = file_son['OMSO2e_003_ColumnAmountSO2']
lon_son = data_obs_son['lon']
so2_son = np.mean(data_obs_son, axis=0)

file_djf = xr.open_dataset('C:/python_work/phd/paper3/phase3_approved/cross_section_phase3/djf.nc')
data_obs_djf = file_djf['OMSO2e_003_ColumnAmountSO2']
lon_djf = data_obs_djf['lon']
so2_djf = np.mean(data_obs_djf, axis=0)

file_mam = xr.open_dataset('C:/python_work/phd/paper3/phase3_approved/cross_section_phase3/mam.nc')
data_obs_mam = file_mam['OMSO2e_003_ColumnAmountSO2']
lon_mam = data_obs_mam['lon']
so2_mam = np.mean(data_obs_mam, axis=0)

file_jja = xr.open_dataset('C:/python_work/phd/paper3/phase3_approved/cross_section_phase3/jja.nc')
data_obs_jja = file_jja['OMSO2e_003_ColumnAmountSO2']
lon_jja = data_obs_jja['lon']
so2_jja = np.mean(data_obs_jja, axis=0)



#WRF
#SON
file1 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_9_sep.npy')
file2 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_10_oct.npy')
file3 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_11_nov.npy')
data_wrf_son = (file1 + file2 + file3)/3
one_d_wrf_son = np.mean(data_wrf_son, axis=0)

#DJF
file4 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_12_dec.npy')
file5 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_1_jan.npy')
file6 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_2_feb.npy')
data_wrf_djf = (file4 + file5 + file6)/3
one_d_wrf_djf = np.mean(data_wrf_djf, axis=0)

#MAM
file7 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_3_mar.npy')
file8 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_4_apr.npy')
file9 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_5_may.npy')
data_wrf_mam = (file7 + file8 + file9)/3
one_d_wrf_mam = np.mean(data_wrf_mam, axis=0)

#JJA
file10 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_6_june.npy')
file11 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_7_july.npy')
file12 = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_8_aug.npy')
data_wrf_jja = (file10 + file11 + file12)/3
one_d_wrf_jja = np.mean(data_wrf_jja, axis=0)




# Define coordinates that match the data
coord = xr.Dataset({'lat': (['lat'], np.arange(-7, 1.2, 0.086)),
                     'lon': (['lon'], np.arange(23.4, 31.6, 0.086))})

#Making the plot
fig=plt.figure(figsize=(10, 10), dpi=500)
mpl.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.gcf().subplots_adjust(hspace=0.27, wspace=0.17)

ax = plt.subplot(2,2,1)
plt.plot(lon_son, so2_son, color='deeppink', label='OMI')
plt.plot(coord['lon'], one_d_wrf_son, color='black', label='WRF-Chem')
plt.axvline(29.23, color='green', linestyle='--')
plt.ylabel('$\mathregular{SO_2}$ VCD  (Dobson Units)')
plt.ylim(0, 0.52)
plt.title('SON')

ax = plt.subplot(2,2,2)
plt.plot(lon_djf, so2_djf, color='deeppink', label='OMI')
plt.plot(coord['lon'], one_d_wrf_djf, color='black', label='WRF-Chem')
plt.axvline(29.23, color='green', linestyle='--')
plt.ylim(0, 0.52)
plt.title('DJF')

ax = plt.subplot(2,2,3)
plt.plot(lon_mam, so2_mam, color='deeppink', label='OMI')
plt.plot(coord['lon'], one_d_wrf_mam, color='black', label='WRF-Chem')
plt.axvline(29.23, color='green', linestyle='--')
plt.ylim(0, 0.52)
plt.ylabel('$\mathregular{SO_2}$ VCD  (Dobson Units)')
plt.xlabel('Longitude ($\mathregular{^o}$E)')
plt.title('MAM')

ax = plt.subplot(2,2,4)
plt.plot(lon_jja, so2_jja, color='deeppink', label='OMI')
plt.plot(coord['lon'], one_d_wrf_jja, color='black', label='WRF-Chem')
plt.axvline(29.23, color='green', linestyle='--')
plt.ylim(0, 0.52)
plt.xlabel('Longitude ($\mathregular{^o}$E)')
plt.title('JJA')
