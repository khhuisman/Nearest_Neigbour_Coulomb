
#Author: Karssien Hero Huisman

import numpy as np
from matplotlib import pyplot as plt
import negf_NN_coulomb as negf_method
import selfconsistent_trapz as st
import Scattering_Region as helical_chain

import currents as currents

def func_energies(ef,
                      H,U,W,dim,
                 GammaL,GammaR,
                betaL,betaR,
                   npoints,npoints_tail,
                   T,Vmax,tol_nintegrand,eta = 1/2,factor_extra = 1):
    '''
    Input:
    Hamiltonian0 = Hamiltonian of the isolated molecule without interactions (U=0)
    U            = interaction strength
    npoints      = number of energy points in window [emin,emax]
    npointstail = number of energy points in window [emin2,emin)
    factor_extra = positive number larger or equal to one. The tolerance "tol_nintegrand" that is used to check whether the integral of G<[i,i] is smaller is divided by this number. This is usefull of the densities deviate a lot from the expetec value of 0.5
    Output:
        energiesreal = list of energies to calculate the Glesser integral with.
        emax = upper bound of intergral
        emin = lowest eigenvalue of Hamiltonian0 - 10 eV
        emin2 = lower bound of integral.    
    '''
    
    
    mu_max = max([ef + Vmax*eta,ef + Vmax*(1-eta)])           ## largest chemical potential
    beta = negf_method.func_beta(T) ## beta
    nislist = 0.5*np.ones((dim,))
        
    #Lower bound: 
    ### At lower bound of integral:
    ### <n>  = \int G+ fi Gammai G- dE ~ \int G+ Gammai G- dE
    ### the fermi functions are approximately 1.
    ### The integrand-size is related to the lowest eigenvalue of "Hamiltonian0":
    evlist = np.linalg.eigh(H)[0]        ##list of eigenvalues
    
    ### Therefore we 'guess' a lowest value:
    emin = np.round(int(10*min(evlist))/10 -10 ,2) 
    
    ### and check if this falls within a tolerance 'tol_nintegrand':
    emin2     = emin - 30
    boundbool = False

    
    print('Estimating lower bound...')
    while boundbool == False:
        
        emin2 = emin2 - 10
        
        
        nlist_min = negf_method.ndensity_listi(np.array([emin2]),nislist,
                        ef, ef,
                      H,U,W,dim,
                 GammaL,GammaR,
                betaL,betaR)[0]

        check_zerob, boundbool = st.check_smaller_tol(nlist_min,tol_nintegrand/factor_extra)
#         print(emin2,check_zerob)
    
  
    
    
    ### Upper Bound:
    ### Due to fermi dirac function, the integrand of <n>:
    ### <n> = \int G+ fi Gammai G- dE ~ \int fi dE 
    ### will fall off quickly to zero near the energy ef + V/2
    ### Therefore the largest energy we need to consider is close to ef + Vmax/2:
    
    emax = mu_max #intial guess emax
    fmax = negf_method.fermi_dirac(np.array([emax]),mu_max,beta)[0] #intial guess efmax
    
    ### We continue our search iteratively
    while fmax >= tol_nintegrand/100:
       
        emax = emax + 2
        fmax = negf_method.fermi_dirac(np.array([emax]),mu_max,beta)[0]
    
    
    
    energies_tail = np.linspace(emin2,emin,npoints_tail)  #In wide band limit the integrand has a long "tail"
    energiesreal_prime = np.linspace( emin, emax,npoints) 
    energiesreal = np.unique(np.concatenate((energies_tail,energiesreal_prime)))
   
    return emin2,emin,emax,energiesreal

def func_converged_and_biassymmetric(Varray_total,nP_total,convgP, nM_total,convgM):
    
    '''
    Input: 
    V_array_total : array of bias voltages. Important! it has to be of the form that:
                    V_array_total + np.flip(V_array_total) = 0
                    
                    
    nP_total nM_total = electron densities for positive, negative magnetization of the lead.
    convgP,convgM     = arrays of booleans indicating if results have converged
   
    Output:
    arrays for converged electron densities and corresponding voltages. Also these are selected on their bias voltage symmetry 
    
    '''
    check_zero,zero_bool = st.check_smaller_tol(Varray_total + np.flip(Varray_total),10**-14)
    assert zero_bool == True, 'Check that input bias voltages satisfy required form. Because now V_array_total + np.flip(V_array_total) = {}'.format(check_zero)
    
    
    array_combined = np.logical_and(
                    np.logical_and( np.flip(convgP),np.flip(convgM)),
                   np.logical_and( convgP,convgM)
                  )

    indexlist = np.where(array_combined ==True)[0]
    
    V_convg = np.array([Varray_total[i] for i in indexlist])
    nP_total_convg = np.array([nP_total[i] for i in indexlist])
    nM_total_convg = np.array([nM_total[i] for i in indexlist])
    
    return V_convg,nP_total_convg,nM_total_convg
    

def func_converged(Varray_total,nP_total,convgP, nM_total,convgM):
    
    '''
    Input: 
    V_array_total : array of bias voltages. Important! it has to be of the form that:
                    V_array_total + np.flip(V_array_total) = 0
                    
                    
    nP_total nM_total = electron densities for positive, negative magnetization of the lead.
    convgP,convgM     = arrays of booleans indicating if results have converged
   
    Output:
    arrays for converged electron densities and corresponding voltages.
    
    '''
    array_combined = np.logical_and( convgP,convgM)

    indexlist = np.where(array_combined ==True)[0]
    
    V_convg = np.array([Varray_total[i] for i in indexlist])
    nP_total_convg = np.array([nP_total[i] for i in indexlist])
    nM_total_convg = np.array([nM_total[i] for i in indexlist])
    
    return V_convg,nP_total_convg,nM_total_convg


def func_V0(V_convg):
    index_V0 = np.where(V_convg == 0.0)[0]
    if len(index_V0) == 1:
        V0 = V_convg[index_V0[0]]
        assert V0 == 0.0,'Voltage is not equal to zero'
        return index_V0[0]
    
    
    if len(index_V0) == 0:
        print('No index found for V=0 ')

        
def check_onsager(tol, 
                    HP,HM,
                      U,W,
                      nMconvg,nPconvg,Vconvg,
                      dim,
                    GammaLP,GammaLM,
                    GammaR):
    
    energies = np.linspace(-10,10)
    index_V0 = func_V0(Vconvg)
    TRLP = negf_method.TRL_func(energies, 
                        HP,U,W,nPconvg[index_V0],dim,
                        GammaLP,
                        GammaR)
    TLRM = negf_method.TLR_func(energies, 
                            HM,U,W,nMconvg[index_V0],dim,
                            GammaLM,
                            GammaR)
    deltaT = TRLP-TLRM
    
    zerolist,zerobool = st.check_smaller_tol(deltaT,tol)
    
    assert zerobool == True,'Onsager Reciprocity is not satisfied.'
    plt.title('Transmissions at Vbias = {}'.format(Vconvg[index_V0]))
    plt.plot(energies,deltaT,label = '$T_{LR}(m) - T_{RL}(-m)$')
    plt.xlabel('energies',size = 18)
    plt.ylabel('transmission',size = 18)
    plt.legend(fontsize = 18)
    plt.show()
    
    return zerobool
    
    
def plot_currents(V_convg,IP_array,IM_array):
    deltaI = IP_array-IM_array
    Vprime,PC_array = currents.func_PC_list(IP_array,IM_array,V_convg)
    size = 20
    
    plt.plot(V_convg,IP_array,label = '$I(m)$')
    plt.plot(V_convg,IM_array,label = '$I(-m)$')
    plt.xlabel('Bias Voltage [eV]',size=size)
    plt.ylabel('Current',size=size)
    plt.legend(fontsize = size)
    plt.show()

    plt.plot(V_convg,deltaI,label = '$I(m) - I(-m)$')
    plt.xlabel('Bias Voltage [eV]',size=size)
    plt.ylabel('Magnetocurrent',size=size)
    plt.legend(fontsize = size)
    plt.ticklabel_format(style="sci", scilimits=(0,0),axis = 'y')
    plt.show()

    plt.plot(Vprime,PC_array,label = '$[I(m) - I(-m)]/[I(m) + I(-m)]$')
    plt.xlabel('Bias Voltage [eV]',size=size)
    plt.ylabel('[%]',size=size)
    plt.legend(fontsize = size-5)
    plt.ticklabel_format(style="sci", scilimits=(0,0),axis = 'y')
    plt.show()
    
def plot_electron_densities(Varray_total,nP_total,nM_total):
    size = 20
    plt.plot(Varray_total,nP_total.sum(axis=1) ,label = 'n(m)')
    plt.plot(Varray_total,nM_total.sum(axis=1),label = 'n(-m)')

    plt.xlabel('Bias Voltage [eV]')
    plt.ylabel('Electron Density')
    plt.legend(fontsize = size)
    plt.show()


    plt.plot(Varray_total,nP_total.sum(axis=1) - nM_total.sum(axis=1),label = 'n(m)-n(-m)')

    plt.xlabel('Bias Voltage [eV]')
    plt.ylabel('Electron Density')
    plt.legend(fontsize = size)
    plt.show()
    plt.plot(Varray_total,nP_total-nM_total)

    plt.xlabel('Bias Voltage [eV]')
    plt.ylabel('Electron Density')
    
    plt.show()
    
    
    


### find point where total electron density goes through half-filling
def func_converged_density(n_array,convg_array,ef_array):
    index_converged = np.where(np.array(convg_array) == True)[0]
    n_list_ef_convg       = np.array([n_array[i] for i in index_converged])
    ef_array_convg       = np.array([ef_array[i] for i in index_converged])
    
    return ef_array_convg,n_list_ef_convg


def func_find_clost_values(n_list_ef_c,ef_array_c,y):
    
    '''
    Returns coordinates for which densities cross a threshold y
    
    Input:
    n_list_ef_c = list of electron densities as function of fermi_levels
    ef_array_c = list of fermi levels
    y = value that is crossed
    
    Ouput
    
    ef_min,ef_max,n_min,n_max : x and y coordinates to the left and right of point (x,y).
    '''
    ntotal = np.array(n_list_ef_c).sum(axis = 1)
    n_normalized = ntotal-y


    sign_array = np.sign(n_normalized)
    sign_positive = np.where(sign_array == 1)[0]
    sign_negative = np.where(sign_array == -1)[0]

    index_min = sign_negative[-1]
    index_max = sign_positive[0]


    ef_min = ef_array_c[index_min] 
    ef_max = ef_array_c[index_max] 

    n_min = np.array(n_list_ef_c[index_min]).sum()
    n_max = np.array(n_list_ef_c)[index_max].sum()
    
    return ef_min,ef_max,n_min,n_max


##### Use interpolation to approximate where total electron density is "dim/2"
def interpolation(y2,y1,x2,x1,y):
    a = (y2-y1)/(x2-x1)
    c = y1 - a*x1
    
    
    x = (y-c)/a
    
    return x


def interpolation_all(y,n_array,convg_array,ef_array):
    
    ef_array_c,n_list_ef_c = func_converged_density(n_array,convg_array,ef_array)
    ef_min,ef_max,n_min,n_max = func_find_clost_values(n_list_ef_c,ef_array_c,y)
    ef_half_filling = interpolation(n_max,n_min,ef_max,ef_min,y)
    
    return ef_half_filling