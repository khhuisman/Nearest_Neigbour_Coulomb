#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman


import numpy as np 
from matplotlib import pyplot as plt


############################################################################################
############################################################################################
####################### Fermi - Dirac Function distrubtions & Derivatives ##################
############################################################################################
############################################################################################

def func_beta(T):
    kB = 8.617343*(10**(-5)) # eV/Kelvin
    ''' Input:
    -Temperature in Kelvin
    Output:
    - beta in eV^-1
    '''
    if T > 0:
        return 1/(kB*T)
    if T ==0:
        return np.infty
    if T<0:
        print('Temperature cannot be negative')

# Fermi-Dirac function
def fermi_dirac(energies,mui,beta):
    '''Input:
    - energy of the electron
    - mui = chemical potential
    - beta = 1/(kB*T) with T the temperature and kB the Boltzmann constant
    Output:
    - The fermi-Dirac distribution for energy 
    '''
    
    return  1/(np.exp(beta*(energies-mui) ) + 1 )

############################################################################################
####################### functions concerining electron densities
############################################################################################




def swap_spin(nislist,dim):
    '''
    Returns list of electron densities, summed over nearest neigbours. 
    
    --------------
    
    Input:
    dim   = dimension of hamiltonian modulo spin degrees of freedom.
    nislist = list of elctron densities for site i with spin s: [n1i,n1d,...,niup,nidown].
    Return:
    list of electron densities summed over nearest neigbours.
    '''
        
    assert dim % 2 ==0,'dimension of Hamiltonian must be even.'
    nisbarlist = np.zeros((dim,))
    nisbarlist[1::2] = nislist[0::2]
    nisbarlist[0::2] = nislist[1::2]
    return nisbarlist



def sum_NN(nlist,dim):
    '''
    Returns list of electron densities, summed over nearest neigbours. 
    
    --------------
    
    Input:
    dim   = dimension of hamiltonian modulo spin degrees of freedom.
    nlist = list of elctron densities for site i: [n1,...,ni].
    Return:
    list of electron densities summed over nearest neigbours.
    '''
    
    
    nlist_p1 = np.zeros((dim,))
    nlist_p1[1:dim-1] = np.roll(nlist,1)[1:dim-1] 

    nlist_m1 = np.zeros((dim,))
    nlist_m1[1:dim-1] = np.roll(nlist,-1)[1:dim-1]



    ntotal = nlist_p1 + nlist_m1
    ntotal[0] = nlist[1]
    ntotal[-1] = nlist[-2]

    return ntotal

def sum_spin(nislist,dim):
    
    '''
    Returns sum of electron densities with spin up, down. 
    
    --------------
    
    Input:
    dim   = dimension of hamiltonian including spin degree of freedom
    nislist = list of elctron densities for site i with spin s: [n1i,n1d,...,niup,nidown].
    Return:
    list of electron densities summed over spin.
    '''
    
    return nislist[0::2] + nislist[1::2]


def sum_NN_spin_dof(nislist,dim):
    '''
    Returns list of electron densities  , summed over nearest neigbours. 
    
    --------------
    
    Input:
    dim   = dimension of Hamiltonian including spin degree of freedom.
    nislist = list of elctron densities for site i with spin s: [n1i,n1d,...,niup,nidown].
    Return:
    list of electron densities summed over nearest neigbours.
    '''
    nlist = sum_spin(nislist,dim)
    nsummed = sum_NN(nlist,len(nlist))
    
    nummed_spin = np.zeros((dim))
    nummed_spin[0::2] = nsummed
    nummed_spin[1::2] = nsummed

    return nummed_spin

#########################################################
############# Green's function
#########################################################


def GRA(energies, 
        H,U,W,nislist,dim,
        GammaL,
        GammaR
         ):
    
    '''
    Returns retarded & advanced Green's functions for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN array)
    
    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
       

    npoints_energy = len(energies)
    array_halves = 0.5*np.ones((dim,))
    
    
    HW = W*np.diag(sum_NN_spin_dof(nislist-array_halves,dim),0)
    HU = U*np.diag(swap_spin(nislist-array_halves,dim),0)

    ### convert input matrices to tensors
    array_ones_energies = np.ones((npoints_energy,))
    identity_matrix = np.identity(dim,dtype =complex)

    Energy_tensor = np.tensordot(energies,identity_matrix,axes =0)
    H_tensor      = np.tensordot(array_ones_energies,H,axes =0)
    HU_tensor     = np.tensordot(array_ones_energies,HU,axes =0)
    HW_tensor     = np.tensordot(array_ones_energies,HW,axes =0)
    
    Gamma_tensor = np.tensordot(array_ones_energies,GammaL + GammaR,axes =0)    
    
    ## Calculate Green's functions
    total = Energy_tensor - H_tensor - HU_tensor - HW_tensor + (1j/2)*Gamma_tensor 
    GR    = np.linalg.inv(total)                            ## retarded Green's function
    
#     GA    = np.linalg.inv(Energy_tensor - H_tensor - HU_tensor - HW_tensor - (1j/2)*Gamma_tensor) 
    GA    = np.transpose(np.conjugate(GR),axes = (0,2,1))   ## advanced Green's function
    
    return GR,GA
    

#########################################################
############# Elastic Transmissions & currents
#########################################################
    
# Transmission left to right
# Only valid for 2terminal junctions
def TLR_func(energies, 
        H,U,W,nislist,dim,
        GammaL,
        GammaR):
    
    '''
    Returns transmission from left to right for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN array)
    
    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
    
    GR,GA = GRA(energies,H,U,W,nislist,dim,GammaL,GammaR)
    
    npoints_energy = len(energies)
    array_ones_energies = np.ones((npoints_energy,))
    GammaL_tensor = np.tensordot(array_ones_energies, GammaL,axes =0)
    GammaR_tensor = np.tensordot(array_ones_energies, GammaR,axes =0)


    T = np.matmul(np.matmul(np.matmul(GammaL_tensor,GA),GammaR_tensor),GR)
    TLRe = np.matrix.trace(T,axis1 = 1,axis2=2).real


    return TLRe



# Transmission left to right
# Only valid for 2terminal junctions
def TRL_func(energies, 
        H,U,W,nislist,dim,
        GammaL,
        GammaR):
    
    '''
    Returns transmission from right to left for every energy in energies
    
    ---------
    Parameters :
    
    energies = array of energies
    H        = The Hamiltonian of the isolated system (NXN array)
    dim      = N (first/second component of the shape of the Hamiltonian.)
    GammaL   = Gamma matrix of left lead  (NXN array)
    GammaR   = Gamma matrix of right lead (NXN array)
    
    Returns:
    
    GR,GA = Retarded & Advanced Green's function for all elements in energies
    
    '''
    
    GR,GA = GRA(energies,H,U,W,nislist,dim,GammaL,GammaR)
    
    npoints_energy = len(energies)
    array_ones_energies = np.ones((npoints_energy,))
    GammaL_tensor = np.tensordot(array_ones_energies,GammaL ,axes =0)
    GammaR_tensor = np.tensordot(array_ones_energies, GammaR,axes =0)

    T = np.matmul(np.matmul(np.matmul(GammaR_tensor,GA),GammaL_tensor),GR)
    TRLe = np.matrix.trace(T,axis1 = 1,axis2=2).real


    return TRLe



def integrand_current(energies,nislist,muL,muR, H,U,W,dim,
                             GammaL,GammaR,
                          betaL,betaR):
    
    '''
    Input:
    - energies = array of energies of incoming electron.
    - betaL,betaR = the beta = 1/(kB T) of the left,right lead
    - muL,muR = chemical potential of left,right lead

    Output:
    - Current calculated with Landauer-Buttiker Formula '''
        
    
    fL = fermi_dirac(energies,muL,betaL)
    fR = fermi_dirac(energies,muR,betaR)
    
    
    TLR = TLR_func(energies,H,U,W,nislist,dim,GammaL,GammaR)
    integrand = TLR*(fL-fR)
    
    return integrand




################################################################## 
#############  Quantities related to Electron Densities   
##################################################################


########### Sigma's ###########

def SigmaLesser(energies,
                GammaL,GammaR,
                muL , muR,
                betaL,betaR):
    
    shape = GammaL.shape
    SigmaLess = np.zeros(shape)
    
    fL = fermi_dirac(energies,muL,betaL)
    fR = fermi_dirac(energies,muR,betaR)
    
    GammaLfL = np.tensordot(fL,GammaL,axes =0)
    GammaRfR = np.tensordot(fR,GammaR,axes =0)
    
    SigmaLess = np.multiply(1j,
                                np.add(
                                        GammaLfL, 
                                        GammaRfR
                                      )
                           )
    
    return SigmaLess



def SigmaGreater(energies,
                GammaL,GammaR,
                muL , muR,
                betaL,betaR):
    
   
    fL = fermi_dirac(energy,muL,betaL)
    fR = fermi_dirac(energy,muR,betaR)
    
    GammaLfLm = np.tensordot(1-fL,GammaL,axes =0)
    GammaRfRm = np.tensordot(1-fR,GammaR,axes =0)
    
    SigmaGreat = np.multiply(-1j,np.add(GammaLfLm,
                                        GammaRfRm
                                      )
                                
                           )
    
    
    shape_identity = GammaL.shape
#     iden = np.multiply(2,np.identity(shape_identity[0]))
    iden = np.add(GammaL,GammaR)
    
    # alpha = lead label
    # Equation (2.123) from Seldenthuys thesis:
        #Sigma>_alpha = -1j Gamma_alpha(1- f_alpha) 
        #Sigma>       = Sigma>_L + Sigma>_R
        #Sigma>       = -1j (GammaL + GammaR - GammaL*fL - GammaR*fR)
    SigmaGreat = np.multiply(-1j,np.add(iden,
                                       np.add(
                                         np.multiply(GammaL,-fL), 
                                         np.multiply(GammaR,-fR)
                                              )
                                      )
                                
                           )
    
    return SigmaGreat
    
########### Lesser Green's Function ###########


def GLesser(energy,H,U,W,nislist,dim,
                GammaL,GammaR,
                muL, muR,
                betaL,betaR):
    
    GR,GA = GRA(energy,
                     H,U,W,nislist,dim,
                     GammaL,GammaR)
    
    SigmaLess = SigmaLesser(energy,GammaL,GammaR,
                            muL, muR ,
                            betaL,betaR)
    
    
    
    
    # G< = GR.Sigma<.GA
    Gless = np.matmul(GR,
                        np.matmul(SigmaLess,GA)
                     )
    
    return Gless



########### Greater Green's Function ###########

def GGreater(energies,H,U,W,nislist,dim,
                GammaL,GammaR,
                muL, muR,
                betaL,betaR):
    
    GR,GA = GRA(energies,
                     H,U,W,nislist,dim,
                     GammaL,GammaR)
    
    SigmaGreat = SigmaGreater(energies,
                              GammaL,GammaR,
                            muL, muR ,
                            betaL,betaR)
    
    
    
    
    # G< = GR.Sigma<.GA
    GG = np.matmul(GR,
                        np.matmul(SigmaGreat,GA)
                     )
    
    return GG


########### Density of states ###########

def density_of_states(energies,
                     H,U,W,nislist,dim,
                      GammaL,GammaR):
    
    GR,GA = GRA(energies,
                      H,U,W,nislist,dim,
                     GammaL,GammaR)
    
    A = 1j*np.subtract(GR,GA)
    
    DOS = np.matrix.trace(A,axis1 = 1,axis2=2)/(2*np.pi) 
    
    return DOS





def ndensity_listi(energies,nislist,
                        muL, muR,
                      H,U,W,dim,
                 GammaL,GammaR,
                betaL,betaR):
    
    
    '''
    Input:
    - energy = energy
    - ilist = list of indices for which one want to calculate the electron density.
    - GammaL,GammR = Gamma Matrices of left,right lead.
    - muL,muR = chemical potential of the left,right lead.
    - betaL,betaR = beta = (kBT)^-1 of left,right lead.
    Ouput
    - List of electron densities on the molecule.
    '''
    
    
    Gless = GLesser(energies,
                      H,U,W,nislist,dim,
                    GammaL,GammaR,
                muL, muR,
                betaL,betaR)
    
    
    nij_arrayE = np.multiply(-1j/(2*np.pi),Gless)
    nii_arrayE = np.diagonal(nij_arrayE, offset=0, axis1=1, axis2=2).real
    
    return nii_arrayE

def integrate_electrondensities(energies,
                      H,U,W,nislist,dim,
                 GammaL,GammaR,
                muL, muR,
                betaL,betaR):
    
    
    '''
    Input:
    - energy = energy
    - ilist = list of indices for which one want to calculate the electron density.
    - GammaL,GammR = Gamma Matrices of left,right lead.
    - muL,muR = chemical potential of the left,right lead.
    - betaL,betaR = beta = (kBT)^-1 of left,right lead.
    Ouput
    - List of electron densities on the molecule.
    '''
    
    
    nii_arrayE =ndensity_listi(energies,nislist,
                        muL, muR,
                      H,U,W,dim,
                 GammaL,GammaR,
                betaL,betaR)
    
   
    
    return np.trapz(nii_arrayE,energies,axis =0)


