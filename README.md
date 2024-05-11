# Astro5470_Project

## Overview of rad_trans.py
Functions that will help in computing the radiative tranfer model for the selected molecule

Q(T, T_list, log_Q)\
sigma(rest_freq, FWHM)\
vel_shift(vlsr, rest_freq)\
line_profile(freq, rest_freq, FWHM, vlsr)\
opt_depth(freq, Ntot, T, T_list, log_Q, rest_freq, FWHM, vlsr, Aij, gu, Eup)\
Jv(T, freq)\
beam_dilution(source_size, bmax, bmin)\
brightness_temp(freq, Ntot, T, FWHM, vlsr, param)\
rad_trans(dbname, molname, freq, Ntot, T, FWHM, vlsr, size=1)\


## Overview of read_lines.py
A set of function meant to read in spectroscopic data from the CDMS and JPL databases. These function will also convert the values of the databases into other parameters needed to compute the radiative transfer model.

List of functions included:\
read_JPLQ(mol_name)\
read_JPL(mol_name, min_freq, max_freq)\
read_CDMSQ(tag)\
read_CDMS(mol_name, min_freq, max_freq)\


## Overview of miscell.py
A set of function that will help in plotting a rotational diagram with some other additional function that are helpful in getting line paramters

List of functions included:\
gaussian(x, a, b, c)\
inv_gaussian(y, a, b, c)\
get_FWHM(param)\
get_vel_shift(freq, rest_freq)\
Nup(rest_freq, Aij, integ_temp)\
area_under_gauss(param)\
ln_Nu_gu(rest_freq, Aij, gup, tot_area)\

For more information on the functions, refer to the wiki