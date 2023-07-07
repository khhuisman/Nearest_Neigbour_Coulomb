
#Author: Karssien Hero Huisman

import numpy as np
from matplotlib import pyplot as plt


#### Geometrical Parameters of SOC

def rvec(m,a,c,M,N,chirality):
    
    '''
    Input:
    - N = number of sites per lap ( 1 lap = rotation by 2pi around helix axis)
    - M = number of laps
    - c = length of the molecule
    - a = radius
    - m = site label
    - chirality = boolean indicating the chirality of the helix.
    Ouput:
    - positional vector of helix
    
    '''
    if chirality==True:
        sign = 1
    if chirality== False:
        sign = -1
        
    phim = 2*np.pi*(m-1)/N # Fransson
    Rpos = [a*np.cos(phim),a*np.sin(sign*phim),c*(m-1)/(M*N-1)] # Fransson


    return Rpos






def dvec(m,stilde,a,c,M,N,chirality):
    
    '''
    Input:
    - N = number of sites per lap ( 1 lap = rotation by 2pi around helix axis)
    - M = Number of laps
    - c = length of the molecule
    - a = radius
    - m = site label
    - stilde = integer, for a site 1,2,3,.. next to site m 
    - chirality = boolean indicating the chirality of the helix.
    Ouput:
    - vector D(m+s): difference between positional vector:  on site m and m+s.
    '''
    
    Rm = rvec(m,a,c,M,N,chirality)
    Rms = rvec(m+stilde,a,c,M,N,chirality)
    
    dRmRms = np.subtract(Rm,Rms)
    norm = np.linalg.norm(dRmRms)
    
    Dvector = np.multiply(1/norm,dRmRms)
    
    return Dvector


def Vmvec(m,stilde,a,c,M,N,chirality=True):
    
    '''
    Input:
    - N = number of sites per lap ( 1 lap = rotation by 2pi around helix axis)
    - M = Number of laps
    - c = length of the molecule
    - a = radius
    - m = site label
    - stilde = integer, for a site 1,2,3,.. next to site m
    - chirality = boolean indicating the chirality of the helix.
    Ouput:
    -  cross procducht between D(m+s) and D(m+2s).
    '''
    
    dms = dvec(m,stilde,a,c,M,N,chirality)
    dm2s = dvec(m,2*stilde,a,c,M,N,chirality) # FRANSSON
    
    vvec = np.cross(dms,dm2s)
    
    return vvec
