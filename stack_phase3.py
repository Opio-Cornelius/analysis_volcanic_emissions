# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 23:41:34 2022

@author: opio
"""

import numpy as np


" Model data "

#2009
md_2009_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2009_9_sep.npy')
md_2009_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2009_10_oct.npy')
md_2009_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2009_11_nov.npy')
md_2009_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2009_12_dec.npy')

#2010
md_2010_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_1_jan.npy')
md_2010_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_2_feb.npy')
md_2010_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_3_mar.npy')
md_2010_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_4_apr.npy')
md_2010_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_5_may.npy')
md_2010_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_6_june.npy')
md_2010_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_7_july.npy')
md_2010_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_8_aug.npy')
md_2010_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_9_sep.npy')
md_2010_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_10_oct.npy')
md_2010_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_11_nov.npy')
md_2010_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2010_12_dec.npy')


#2011
md_2011_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_1_jan.npy')
md_2011_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_2_feb.npy')
md_2011_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_3_mar.npy')
md_2011_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_4_apr.npy')
md_2011_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_5_may.npy')
md_2011_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_6_june.npy')
md_2011_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_7_july.npy')
md_2011_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_8_aug.npy')
md_2011_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_9_sep.npy')
md_2011_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_10_oct.npy')
md_2011_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_11_nov.npy')
md_2011_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2011_12_dec.npy')


#2012
md_2012_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_1_jan.npy')
md_2012_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_2_feb.npy')
md_2012_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_3_mar.npy')
md_2012_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_4_apr.npy')
md_2012_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_5_may.npy')
md_2012_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_6_june.npy')
md_2012_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_7_july.npy')
md_2012_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_8_aug.npy')
md_2012_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_9_sep.npy')
md_2012_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_10_oct.npy')
md_2012_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_11_nov.npy')
md_2012_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2012_12_dec.npy')


#2013
md_2013_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_1_jan.npy')
md_2013_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_2_feb.npy')
md_2013_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_3_mar.npy')
md_2013_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_4_apr.npy')
md_2013_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_5_may.npy')
md_2013_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_6_june.npy')
md_2013_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_7_july.npy')
md_2013_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_8_aug.npy')
md_2013_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_9_sep.npy')
md_2013_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_10_oct.npy')
md_2013_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_11_nov.npy')
md_2013_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2013_12_dec.npy')


#2014
md_2014_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_1_jan.npy')
md_2014_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_2_feb.npy')
md_2014_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_3_mar.npy')
md_2014_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_4_apr.npy')
md_2014_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_5_may.npy')
md_2014_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_6_june.npy')
md_2014_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_7_july.npy')
md_2014_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_8_aug.npy')
md_2014_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_9_sep.npy')
md_2014_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_10_oct.npy')
md_2014_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_11_nov.npy')
md_2014_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2014_12_dec.npy')


#2015
md_2015_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_1_jan.npy')
md_2015_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_2_feb.npy')
md_2015_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_3_mar.npy')
md_2015_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_4_apr.npy')
md_2015_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_5_may.npy')
md_2015_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_6_june.npy')
md_2015_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_7_july.npy')
md_2015_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_wrf/wrf_2015_8_aug.npy')



model_tr = np.stack((md_2009_sep, md_2009_oct, md_2009_nov, md_2009_dec, md_2010_jan, md_2010_feb, md_2010_mar,
                     md_2010_apr, md_2010_may, md_2010_june, md_2010_july, md_2010_aug, md_2010_sep, md_2010_oct,
                     md_2010_nov, md_2010_dec, md_2011_jan, md_2011_feb, md_2011_mar, md_2011_apr, md_2011_may, 
                     md_2011_june, md_2011_july, md_2011_aug, md_2011_sep, md_2011_oct, md_2011_nov, md_2011_dec,
                     md_2012_jan, md_2012_feb, md_2012_mar,md_2012_apr, md_2012_may, md_2012_june, md_2012_july, 
                     md_2012_aug, md_2012_sep, md_2012_oct, md_2012_nov, md_2012_dec, md_2013_jan, md_2013_feb, 
                     md_2013_mar,md_2013_apr, md_2013_may, md_2013_june, md_2013_july, md_2013_aug))

                     
model_v = np.stack((md_2013_sep, md_2013_oct, md_2013_nov, md_2013_dec, md_2014_jan, md_2014_feb, md_2014_mar, 
                    md_2014_apr, md_2014_may, md_2014_june, md_2014_july, md_2014_aug))

model_tst = np.stack((md_2014_sep, md_2014_oct, md_2014_nov, md_2014_dec, md_2015_jan, md_2015_feb, md_2015_mar, 
                      md_2015_apr, md_2015_may, md_2015_june, md_2015_july, md_2015_aug))


np.save('C:/python_work/phd/paper3/phase3_approved/stacked_data/so2_model_train_phase3', model_tr)
np.save('C:/python_work/phd/paper3/phase3_approved/stacked_data/so2_model_validate_phase3', model_v)
np.save('C:/python_work/phd/paper3/phase3_approved/stacked_data/so2_model_test_phase3', model_tst)






" Observation data "

#2009
ob_2009_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2009_9_sep.npy')
ob_2009_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2009_10_oct.npy')
ob_2009_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2009_11_nov.npy')
ob_2009_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2009_12_dec.npy')

#2010
ob_2010_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_1_jan.npy')
ob_2010_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_2_feb.npy')
ob_2010_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_3_mar.npy')
ob_2010_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_4_apr.npy')
ob_2010_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_5_may.npy')
ob_2010_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_6_june.npy')
ob_2010_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_7_july.npy')
ob_2010_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_8_aug.npy')
ob_2010_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_9_sep.npy')
ob_2010_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_10_oct.npy')
ob_2010_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_11_nov.npy')
ob_2010_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2010_12_dec.npy')


#2011
ob_2011_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_1_jan.npy')
ob_2011_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_2_feb.npy')
ob_2011_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_3_mar.npy')
ob_2011_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_4_apr.npy')
ob_2011_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_5_may.npy')
ob_2011_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_6_june.npy')
ob_2011_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_7_july.npy')
ob_2011_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_8_aug.npy')
ob_2011_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_9_sep.npy')
ob_2011_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_10_oct.npy')
ob_2011_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_11_nov.npy')
ob_2011_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2011_12_dec.npy')


#2012
ob_2012_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_1_jan.npy')
ob_2012_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_2_feb.npy')
ob_2012_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_3_mar.npy')
ob_2012_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_4_apr.npy')
ob_2012_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_5_may.npy')
ob_2012_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_6_june.npy')
ob_2012_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_7_july.npy')
ob_2012_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_8_aug.npy')
ob_2012_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_9_sep.npy')
ob_2012_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_10_oct.npy')
ob_2012_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_11_nov.npy')
ob_2012_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2012_12_dec.npy')


#2013
ob_2013_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_1_jan.npy')
ob_2013_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_2_feb.npy')
ob_2013_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_3_mar.npy')
ob_2013_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_4_apr.npy')
ob_2013_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_5_may.npy')
ob_2013_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_6_june.npy')
ob_2013_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_7_july.npy')
ob_2013_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_8_aug.npy')
ob_2013_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_9_sep.npy')
ob_2013_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_10_oct.npy')
ob_2013_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_11_nov.npy')
ob_2013_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2013_12_dec.npy')


#2014
ob_2014_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_1_jan.npy')
ob_2014_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_2_feb.npy')
ob_2014_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_3_mar.npy')
ob_2014_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_4_apr.npy')
ob_2014_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_5_may.npy')
ob_2014_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_6_june.npy')
ob_2014_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_7_july.npy')
ob_2014_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_8_aug.npy')
ob_2014_sep = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_9_sep.npy')
ob_2014_oct = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_10_oct.npy')
ob_2014_nov = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_11_nov.npy')
ob_2014_dec = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2014_12_dec.npy')


#2015
ob_2015_jan = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2015_1_jan.npy')
ob_2015_feb = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2015_2_feb.npy')
ob_2015_mar = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2015_3_mar.npy')
ob_2015_apr = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2015_4_apr.npy')
ob_2015_may = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2015_5_may.npy')
ob_2015_june = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2015_6_june.npy')
ob_2015_july = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2015_7_july.npy')
ob_2015_aug = np.load('C:/python_work/phd/paper3/phase3_approved/so2_output_phase3/so2_obs/obs_2015_8_aug.npy')



obs_tr = np.stack((ob_2009_sep, ob_2009_oct, ob_2009_nov, ob_2009_dec, ob_2010_jan, ob_2010_feb, ob_2010_mar,
                     ob_2010_apr, ob_2010_may, ob_2010_june, ob_2010_july, ob_2010_aug, ob_2010_sep, ob_2010_oct,
                     ob_2010_nov, ob_2010_dec, ob_2011_jan, ob_2011_feb, ob_2011_mar, ob_2011_apr, ob_2011_may, 
                     ob_2011_june, ob_2011_july, ob_2011_aug, ob_2011_sep, ob_2011_oct, ob_2011_nov, ob_2011_dec,
                     ob_2012_jan, ob_2012_feb, ob_2012_mar,ob_2012_apr, ob_2012_may, ob_2012_june, ob_2012_july, 
                     ob_2012_aug, ob_2012_sep, ob_2012_oct, ob_2012_nov, ob_2012_dec, ob_2013_jan, ob_2013_feb, 
                     ob_2013_mar,ob_2013_apr, ob_2013_may, ob_2013_june, ob_2013_july, ob_2013_aug))

                     
obs_v = np.stack((ob_2013_sep, ob_2013_oct, ob_2013_nov, ob_2013_dec, ob_2014_jan, ob_2014_feb, ob_2014_mar, 
                    ob_2014_apr, ob_2014_may, ob_2014_june, ob_2014_july, ob_2014_aug))

obs_tst = np.stack((ob_2014_sep, ob_2014_oct, ob_2014_nov, ob_2014_dec, ob_2015_jan, ob_2015_feb, ob_2015_mar, 
                      ob_2015_apr, ob_2015_may, ob_2015_june, ob_2015_july, ob_2015_aug))


np.save('C:/python_work/phd/paper3/phase3_approved/stacked_data/so2_obs_train_phase3', obs_tr)
np.save('C:/python_work/phd/paper3/phase3_approved/stacked_data/so2_obs_validate_phase3', obs_v)
np.save('C:/python_work/phd/paper3/phase3_approved/stacked_data/so2_obs_test_phase3', obs_tst)