import numpy as np
from scipy.optimize import curve_fit
from astropy import constants as const
import astropy.units as u


h = const.h.value    # J/s
c = const.c.value    # m/s
k = const.k_B.value  # J/K


def Q(T, T_list, log_Q):
    """ Compute Partition function at given temperature via interpolation"""

    nan_ind = np.where(np.isnan(log_Q))

    T_list = np.delete(T_list, nan_ind)
    log_Q = np.delete(log_Q, nan_ind)

    log_T = np.log10(T_list)

    def line(x, m, b):
        return m*x + b

    if T >= T_list[-1]:
        param, _ = curve_fit(line, log_T[-2:], log_Q[-2:])
        return 10**(param[0]*np.log10(T) + param[1])

    else:
        index_min = np.where(log_T <= np.log10(T))[0]
        index_max = np.where(log_T > np.log10(T))[0]

        ind = np.array([len(index_min)-1 , index_max[0]])
        
        param, _ = curve_fit(line, log_T[ind], log_Q[ind])

        return 10**(param[0]*np.log10(T) + param[1])

def sigma(rest_freq, FWHM):
    """ Return the line width of gaussian in freq units
        The input FWHM needs to be input in velocity units"""
    return (rest_freq / (c*np.sqrt(8*np.log(2)))) * FWHM


def vel_shift(vlsr, rest_freq):
    """ Return diff btwn rest_freq and vel_shifted (observed) freq """
    
    obs_freq = (rest_freq * (1 - (vlsr/c)))  # input in m/s
    delta_freq = rest_freq - obs_freq
    
    return delta_freq  
    
    
def line_profile(freq, rest_freq, FWHM, vlsr):
    """ Return the line profile """
    A = (1 / (sigma(rest_freq, FWHM) * np.sqrt(2*np.pi)))
    B = -(freq - (rest_freq + vel_shift(vlsr, rest_freq)))**2 / (2*sigma(rest_freq, FWHM)**2)

    return A*np.exp(B)


def opt_depth(freq, Ntot, T, T_list, log_Q, rest_freq, FWHM, vlsr, Aij, gu, Eup):
    """ Compute the Optical Depth """
    
    component1 = (c**2/(8*np.pi*(freq + vel_shift(vlsr, rest_freq))**2))
    component2 = (Ntot/Q(T, T_list, log_Q))
    component3 = Aij*gu* np.exp(-Eup/T)
    component4 = (np.exp((h*rest_freq)/(k*T)) - 1)
    profile = line_profile(freq, rest_freq, FWHM, vlsr)
    
    return component1 * component2 * component3 * component4 * profile
    
    
def Jv(T, freq):
    return ((h*freq)/k) * (1 / (np.exp((h*freq)/(k*T)) - 1))


def beam_dilution(source_size, bmax, bmin):
    """ Compute the Beam Dilution Factor """
    return source_size**2 / (source_size**2 + (bmax*bmin))


def brightness_temp(freq, Ntot, T, FWHM, vlsr, param):
    """ Compute the Radiative Transfer Equation with output as Brightness Temperature """

    T_list, log_Q, rest_freq, Aij, gu, Eup, back_T, source_size, bmax, bmin = param
    J = Jv(T, rest_freq)
    J_cmb = Jv(2.726, rest_freq)
    tau = opt_depth(freq, Ntot, T, T_list, log_Q, rest_freq, FWHM, vlsr, Aij, gu, Eup)
    bdf = beam_dilution(source_size, bmax, bmin)
    
    return  bdf * (J - J_cmb - back_T)*(1 - np.exp(-tau))
    

def rad_trans(dbname, molname, freq, Ntot, T, FWHM, vlsr, size=1):
    """ Compute the radiative transfer model for all lines belonging to a molecule"""

    # Read in lines and partition function info from slected database
    if dbname == 'CDMS':
        lines = read_CDMS(molname, (min(freq)*u.MHz).to(u.GHz).value, (max(freq)*u.MHz).to(u.GHz).value)   # Read in all lines in spectra range
        T_list, log_Q = read_CDMSQ(molname.split()[0])
    elif dbname == 'JPL':
        lines = read_JPL(molname, (min(freq)*u.MHz).to(u.GHz).value, (max(freq)*u.MHz).to(u.GHz).value)   # Read in all lines in spectra range
        T_list, log_Q = read_JPLQ(molname.split()[1])

    if isinstance(lines, int):
        return 0
    # Filter the lines by removing lines with weak emission
    lines = lines[lines['EUP'] < 400*u.K]    # Remove lines with Eup larger than 400K
    lines = lines[lines['Aij'] > 1e-8*(1/u.s)]
    
    lines_index = np.where((lines['EUP'] > 150*u.K) & (lines['Aij'] < 1e-7*(1/u.s)))
    lines.remove_rows(lines_index)


    # Get the beam size for each line WITHIN THIS SPECTRA ONLY
    beam = np.loadtxt("/Users/olbap/Downloads/beam_min_max.txt")
    spec_bounds = np.loadtxt('spec_bounds.txt')
    
    lines['BMAX'] = np.zeros(len(lines['FREQ']))
    lines['BMIN'] = np.zeros(len(lines['FREQ']))
    
    for i in range(len(spec_bounds)):
        ind = np.where((lines['FREQ'].value > spec_bounds[i,0]) & (lines['FREQ'].value < spec_bounds[i,1]))
        if len(lines['FREQ']) == len(ind[0]):
            lines['BMAX'][ind] = beam[i,0]
            lines['BMIN'][ind] = beam[i,1]


    bmax = lines['BMAX']
    bmin = lines['BMIN']

    rest_freq = lines['FREQ'].value * 1e6  #in Hz
    Eup = lines['EUP'].value  #in K
    Aij = lines['Aij'].value
    gu = lines['GUP'].value
    source_size = size  #in arcsec
    back_T = 5.512
    
    param_list = T_list, log_Q, rest_freq, Aij, gu, Eup, back_T, source_size, bmax, bmin

    Ntot = Ntot *100**2  #in m^-2
    Tex = T  #in Kelvin
    FWHM = FWHM * 1000  #in m/s
    vlsr = vlsr * 1000  #in m/s

    if str(type(freq)) == "<class 'numpy.ndarray'>":
        freq = np.vstack(freq)
        return np.sum(brightness_temp(freq*1e6, Ntot, T, FWHM, vlsr, param_list), axis=1)
    else:
        return np.sum(brightness_temp(freq*1e6, Ntot, T, FWHM, vlsr, param_list))