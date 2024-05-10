import numpy as np
from astropy import constants as const
import astropy.units as u
from astroquery.jplspec import JPLSpec
from astropy.table import QTable
from rad_trans import *
from astroquery.linelists.cdms import CDMS
import sqlite3

def read_JPLQ(mol_name):
    """ Read and return Partition Function values with temp from molecule from JPL """
    
    result = JPLSpec.get_species_table()    # Read in all the partition function values table
    mol = result[result['NAME'] == mol_name]    # Get row of specifiied molecule that contains log(Q)
    temp = mol.meta    # Get the metadata to get the temp for the partion function

    temp = np.array(temp['Temperature (K)'])[::-1]    # Turn temp into a np.array and flip elements

    # Turn the row from mol into np.array of log(Q)
    n = 7
    log_Q = np.ones(n)
    for i in range(n):
        log_Q[n-1-i] = mol['QLOG'+str(i+1)]
    
    return temp, log_Q


def read_JPL(mol_name, min_freq, max_freq):
    """ Read in JPL entries and outputs a table with Eup, Gup, and Aij per rest freq """
    
    response = JPLSpec.query_lines(min_frequency= min_freq * u.GHz,
                                   max_frequency= max_freq * u.GHz,
                                   min_strength=-500,
                                   molecule= mol_name,
                                   get_query_payload=False)
    h = const.h    # J/s
    c = const.c   # m/s
    k = const.k_B # J/K
    
    # Get the partition function values and temps for the computation of Aij
    temp, log_Q = read_JPLQ(mol_name.split()[1])   # string must be split because the mol_name is diff to mol_name of this function

    # Compute the Eup column in J
    response['EUP'] = (response['FREQ'].to(1/u.s) + response['ELO'].to(1/u.m) * c) * h
    response['ELO'] = response['ELO'].to(1/u.m) * c * h

    # Compute the Aij column
    T_0 = 300 * u.K
    Aij_1 = 10**response['LGINT'].value * (2.7964E-16)
    Aij_2 = response['FREQ'].value**2 * (Q(T_0.value, temp, log_Q) / response['GUP'])
    Aij_3 = (np.exp(-response['ELO']/(k*T_0)) - np.exp(-response['EUP']/(k*T_0)))**(-1)
    response['Aij'] = Aij_1 * Aij_2 * Aij_3 * (1/u.s)

    # Convert the Eup to Kelvin
    response['EUP'] = response['EUP']/k
    
    # Create table with the desired columns that will be the output
    output_table = QTable([response['FREQ'], response['EUP'], response['GUP'], response['Aij']],
                          names= ('FREQ', 'EUP', 'GUP', 'Aij'))

    return output_table


def read_CDMSQ(tag):
    """ Read and return Partition Function values with temp from molecule from JPL """
    
    result = CDMS.get_species_table()    # Read in all the partition function values table
    tag = int(tag)    # Turn tag from str to int
    mol = result[result['tag'] == tag]    # Get row of specifiied molecule that contains log(Q)
    
    keys = [k for k in mol.keys() if 'lg' in k]    # Add column names that have 'lg' to a list
    temp = np.array([float(k.split('(')[-1].split(')')[0]) for k in keys])    # Use the keys to get list of temps as floats
    temp_str = np.array([k.split('(')[-1].split(')')[0] for k in keys])    # Same as temp but elements are strings

    # Turn the row from mol into np.array of log(Q)
    n = len(temp)
    log_Q = np.ones(n)
    for i in range(n):
        log_Q[i] = mol['lg(Q('+ str(temp_str[i]) +'))']
    
    return temp[::-1], log_Q[::-1]


def read_CDMS(mol_name, min_freq, max_freq):
    """ Read in JPL entries and outputs a table with Eup, Gup, and Aij per rest freq """
    
    response = CDMS.query_lines(min_frequency= min_freq * u.GHz,
                                   max_frequency= max_freq * u.GHz,
                                   min_strength=-500,
                                   molecule= mol_name,
                                   get_query_payload=False)
    h = const.h    # J/s
    c = const.c   # m/s
    k = const.k_B # J/K

    if response['FREQ'][0] == 'No lines found':
        return 0
    
    # Get the partition function values and temps for the computation of Aij
    temp, log_Q = read_CDMSQ(mol_name.split()[0])   # string must be split because the mol_name is diff to mol_name of this function

    # Compute the Eup column in J
    response['EUP'] = (response['FREQ'].to(1/u.s) + response['ELO'].to(1/u.m) * c) * h
    response['ELO'] = response['ELO'].to(1/u.m) * c * h

    # Compute the Aij column
    T_0 = 300 * u.K
    Aij_1 = 10**response['LGINT'].value * (2.7964E-16)
    Aij_2 = response['FREQ'].value**2 * (Q(T_0.value, temp, log_Q) / response['GUP'])
    Aij_3 = (np.exp(-response['ELO']/(k*T_0)) - np.exp(-response['EUP']/(k*T_0)))**(-1)
    response['Aij'] = Aij_1 * Aij_2 * Aij_3 * (1/u.s)

    # Convert the Eup to Kelvin
    response['EUP'] = response['EUP']/k
    
    # Create table with the desired columns that will be the output
    output_table = QTable([response['FREQ'], response['EUP'], response['GUP'], response['Aij']],
                          names= ('FREQ', 'EUP', 'GUP', 'Aij'))

    return output_table