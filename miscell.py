import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import astropy.units as u
import astropy.constants as const


def gaussian(x, a, b, c):
    return a*np.exp(-b*(x-c)**2)

def inv_gaussian(y, a, b, c):
    """ Inverse of a Gaussian. Input y to get the corresponding x value. """
    return np.sqrt(-(1/b) * np.log(y/a)) + c


def get_FWHM(param):
    """ Compute FWHM 
        Input: gaussian parameters in the form of a list or tuple
        Output: In units of km/s """
    half_max = param[0]/2
    half_max_freq_plus = inv_gaussian(half_max, param[0], param[1], param[2])
    FWHM_vel = ((half_max_freq_plus - param[2])/param[2])*const.c.value*(1/1000)*2

    return FWHM_vel


def get_vel_shift(freq, rest_freq):
    """ Compute the velocity shift of a line using the peak freq
        of a gaussian. Output in units of km/s """
    return ((rest_freq - freq)/rest_freq)*const.c.value*(1/1000)  # in km/s


def Nup(rest_freq, Aij, integ_temp):
    """ Compute the upper column density
        Input: rest freq of line, Aij, and area under gaussian
        Output: units of 1/m^2"""
    return (8*np.pi*const.k_B.value*rest_freq)/(const.h.value*const.c.value**2*Aij) * integ_temp  # in 1/m^2


def area_under_gauss(param):
    """ Get the area under a gaussian from gaussian parameters
        Input: list or tuple of gaussian parameters
        Output: area under gaussian in Hz*K """
    
    # Get standard deviation
    std_dv = np.sqrt(1/(2*param[1]))

    # The lower and upper bounds and step size of integration
    low_bnd = param[2]-3*std_dv
    up_bnd = param[2]+3*std_dv
    width = 0.2441184    # The bin width in the spectral data

    # Compute the freq for the deisred range, and its corresponding intensity
    freq = np.arange(low_bnd, up_bnd, width)
    intense = gaussian(freq, param[0], param[1], param[2])

    # Integrate through the intensity
    tot_area = np.sum(intense)*width*1e6 # Hz K
    
    return tot_area


def ln_Nu_gu(rest_freq, Aij, gup, tot_area):
    """ Compute ln(Nu/gup) in cm^-2 """
    return np.log(Nup(rest_freq*1e6, Aij, tot_area)/gup * (1/100**2))