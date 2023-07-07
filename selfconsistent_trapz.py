#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman


import numpy as np 
from matplotlib import pyplot as plt

def func_Vbias(Vmax,Vnpoints):
    
    dV = Vmax/Vnpoints
    V_pos_bias = np.linspace(dV,Vmax ,Vnpoints)
    Varray_total = np.concatenate((-np.flip(V_pos_bias),np.concatenate((np.array([0]),V_pos_bias))))
    print(Varray_total,len(Varray_total))
    return V_pos_bias,Varray_total





def check_smaller_tol(dn_array,tol):
    
    '''
    Input:
    dn_array = difference between new and old electron density.
    tol = tolearnce
    output = input array where
    
    '''
    
    bool_convg = False
    
    
    sign_array = (np.sign(abs(dn_array) - tol) + 1)*0.5
    an_array_new = sign_array*dn_array
    
    
    len_check = len(np.where(an_array_new==0)[0])
    if len_check == len(dn_array):
        bool_convg = True
    
    return np.round(an_array_new,int(np.ceil(-np.log10(tol)))) ,bool_convg


def calc_electron_density_trapz(energies,nislist,muL, muR,tol_nintegrand,
                                ndensity_energy,*args
                         ):


    '''
    Input:
    - energies = list of energies
    - tol_nintegrand = tolerance placed on integrand of
    - ndensity_energy = function for the electron density at given energy
    - args = other arguments of '' of that function
    Ouput:
    - list of electron densities for site i with spin s 
    '''
    
    nii_arrayE = ndensity_energy(energies,nislist,muL, muR,*args)
    
    #### assert that electron densities tend to zero at bounds of the integral.
    nlist_min = nii_arrayE[0]
    nlist_max = nii_arrayE[-1]
    nlist_minmax = np.concatenate((nlist_min,nlist_max))
    check_zerob, boundbool = check_smaller_tol(nlist_minmax,tol_nintegrand)
    assert boundbool == True, 'Integrand of electron densities is larger then set tol: {}. The values of the integrand are: {}'.format(tol_nintegrand,nlist_minmax)
    

    
    
    ## Return integrated electron densities
    return np.trapz(nii_arrayE,energies,axis =0)



################################################################################
######## Self - Consistent Calculation
################################################################################
# def func_print(an_array,tol):
#     acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    
#     return np.floor(abs(an_array)*10**(acc))/(10**(acc))


# def func_print(an_array,tol):
#     an_array_new = []
#     for element in an_array:
#         if element < tol:
#             an_array_new.append(0)
#         if element >= tol:
#             an_array_new.append(element)
            
#     return np.array(an_array_new)

#### self consistent loop given a voltage
def iteration_linear_mixing(n00_list, muL, muR,
                            max_iteration ,energies,
                            tol,tol_nintegrand,alpha,
                                ndensity_energy,*args
                               ):
    
    '''
    returns iterated electron densities and boolean indicating its convergence
    
    Input
    n00_list = intial guess for loop
    muL,R = chemical potential of left,right lead.
    max_iteration = maximum number of iterations 
    energies = energies to integrate over 
    tol = tolerance on electron densities
    tol_nintegrand = tolerance of integrand of electron densities
    alpha = mixing paramter [0,1]
    ndensity_energy = electron densities as function of energy
    *args = arguments of "ndensity_energy"
    
    also plots densities as funciton of iterations
    
    Ouput:
    n00list_new    = electron densities 
    zero_bool00   = booleans indicating if densities have did  or did not converge ( True,False) respectively.
    
    '''
 
    
 
    k_list = np.arange(max_iteration)
    nk_iterations_list = []
    nk_iterations_list.append(n00_list)
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    
    for k in k_list:
        

        # Calculate relevant electron densities:
        n00list_new =  calc_electron_density_trapz(energies,n00_list,
                                                   muL, muR,
                                                   tol_nintegrand,
                                                    ndensity_energy,*args
                                                  )
        
        nk_iterations_list.append(n00list_new)
        R_array = abs(n00_list-n00list_new)
        
         #check if values have converged
        check_zero00, zero_bool00 = check_smaller_tol(R_array,tol)
        print(check_zero00)
        

        if zero_bool00 == True:
            break

        ###Linear mixing   
        n00_list = (1-alpha)*n00list_new + alpha*n00_list
        
       
        
    return n00list_new,zero_bool00


#### self consistent loop given a voltage
def iteration_linear_mixing_noprint(n00_list, muL, muR,
                            max_iteration ,energies,
                            tol,tol_nintegrand,alpha,
                                ndensity_energy,*args
                               ):
    
    '''
    returns iterated electron densities and boolean indicating its convergence
    
    Input
    n00_list = intial guess for loop
    muL,R = chemical potential of left,right lead.
    max_iteration = maximum number of iterations 
    energies = energies to integrate over 
    tol = tolerance on electron densities
    tol_nintegrand = tolerance of integrand of electron densities
    alpha = mixing paramter [0,1]
    ndensity_energy = electron densities as function of energy
    *args = arguments of "ndensity_energy"
    
    also plots densities as funciton of iterations
    
    Ouput:
    n00list_new    = electron densities 
    zero_bool00   = booleans indicating if densities have did  or did not converge ( True,False) respectively.
    
    '''
 
    
 
    k_list = np.arange(max_iteration)
    nk_iterations_list = []
    nk_iterations_list.append(n00_list)
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    
    for k in k_list:
        

        # Calculate relevant electron densities:
        n00list_new =  calc_electron_density_trapz(energies,n00_list,
                                                   muL, muR,
                                                   tol_nintegrand,
                                                    ndensity_energy,*args
                                                  )
        
        nk_iterations_list.append(n00list_new)
        R_array = abs(n00_list-n00list_new)

        
        #check if values have converged
        check_zero00, zero_bool00 = check_smaller_tol(R_array,tol)
       
        
        if zero_bool00 == True:
            break

        ###Linear mixing   
        n00_list = (1-alpha)*n00list_new + alpha*n00_list
        
       
        
    return n00list_new,zero_bool00


#### self consistent loop given a voltage
def iteration_linear_mixing_plot(n00_list,muL, muR,
                            max_iteration ,energies,
                                 tol,tol_nintegrand,alpha,
                                ndensity_energy,*args
                               ):
    
    '''
    returns iterated electron densities and boolean indicating its convergence
    
    Input
    n00_list = intial guess for loop
    muL,R = chemical potential of left,right lead.
    max_iteration = maximum number of iterations 
    energies = energies to integrate over 
    tol = tolerance on electron densities
    tol_nintegrand = tolerance of integrand of electron densities
    alpha = mixing paramter [0,1]
    ndensity_energy = electron densities as function of energy
    *args = arguments of "ndensity_energy"
    
    also plots densities as funciton of iterations
    
    Ouput:
    n00list_new    = electron densities 
    zero_bool00   = booleans indicating if densities have did  or did not converge ( True,False) respectively.
    
    '''
 
    k_list = np.arange(max_iteration)
    nk_iterations_list = []
    nk_iterations_list.append(n00_list)
    acc = int(np.ceil(-np.log(tol)/np.log(10))) 
    for k in k_list:
        

        # Calculate relevant electron densities:
        n00list_new =  calc_electron_density_trapz(energies,n00_list,
                                                   muL, muR,
                                                   tol_nintegrand,
                                                    ndensity_energy,*args
                                                  )
        
        nk_iterations_list.append(n00list_new)
        R_array = abs(n00_list-n00list_new)
        
#         print(np.round(func_print(R_array,tol), acc))
#         print(np.round(R_array, acc))
        
        #check if values have converged
        check_zero00, zero_bool00 = check_smaller_tol(R_array,tol)
        print(check_zero00)
       
        
        if zero_bool00 == True:
            break

        ###Linear mixing   
        n00_list = (1-alpha)*n00list_new + alpha*n00_list
    
        
    plt.plot(nk_iterations_list)
    plt.show()
        
    return n00list_new,zero_bool00



################################################################################
######## Self - Consistent Calculation: loop over all voltages
################################################################################

def self_consistent_trapz_mixing_in(ef,V_list,
                                  n00_list_guess,
                                  max_iteration,energies,
                                    tol,tol_nintegrand,
                                    alpha,
                                ndensity_energy,*args,plotbool=False,printbool=False):
    
    '''
    returns electron densities after self-consistent calculation with list of booleans if densities have converged.
    
    Input
    
    ef = Fermi level
    V_list = list of bias voltages
    n00_list_guess = intial guess for first in V_list
    max_iteration = maximum number of iterations 
    energies = energies to integrate over 
    tol = tolerance on electron densities
    tol_nintegrand = tolerance of integrand of electron densities
    alpha = mixing paramter [0,1]
    ndensity_energy = electron densities as function of energy
    *args = arguments of "ndensity_energy"
    plotbool = boolean if True plots electron density as function of iterations for every bias voltage
    
    Ouput:
    n_list    = list of electron densities as funciton of bias voltage
    convglist = list of booleans indicating if densities converged (True) or dit not converge(False)
    
    '''
    
    n_list = []
    convglist = []
  
    
    for i in range(len(V_list)):
        V = V_list[i]
        muL,muR = ef +V/2, ef -V/2
        
        
        
        if i ==0:
            n00_list_init = n00_list_guess
            

              

        if i !=0:
                
            n00_list_init = n_list[i-1] #initial guess.
            
        
        if plotbool== True and printbool==True:
            print ('--- V = {} ---'.format(V))
            nlist_k,zero_bool = iteration_linear_mixing_plot(n00_list_init,
                                                             muL, muR,
                                                           max_iteration ,energies,
                                                             tol,tol_nintegrand,
                                                             alpha,
                                                            ndensity_energy,*args
                                   )
        if plotbool== False and printbool == True:
            print ('--- V = {} ---'.format(V))
            nlist_k,zero_bool = iteration_linear_mixing(n00_list_init,
                                                             muL, muR,
                                                           max_iteration ,energies,
                                                             tol,tol_nintegrand,
                                                             alpha,
                                                            ndensity_energy,*args
                                   )
            
        if printbool==False:
            nlist_k,zero_bool = iteration_linear_mixing_noprint(n00_list_init,
                                                             muL, muR,
                                                           max_iteration ,energies,
                                                             tol,tol_nintegrand,
                                                             alpha,
                                                            ndensity_energy,*args
                                   )
        
        n_list.append(nlist_k)
        convglist.append(zero_bool)
        
        
    return n_list,convglist






def self_consistent_trapz_PN(ef,V_list_pos_bias,n00_V0_guess,
                                  max_iteration,energies,
                                    tol,tol_nintegrand,
                                    alpha,
                                ndensity_energy,*args,
                             plotbool=False,printbool=True):
    
    for Vbias in V_list_pos_bias:
        assert Vbias >0,'Biases in input list must greater than zero.'
    
    V_list_neg_bias = -1*V_list_pos_bias
    
                                  
    nV0_list,convglistV0 =  self_consistent_trapz_mixing_in(ef,[0],
                                  n00_V0_guess,
                                  max_iteration,energies,
                                    tol,tol_nintegrand,
                                    alpha,
                            ndensity_energy,*args,plotbool=plotbool,printbool=printbool)
    
    #Sweep for positive and negative bias voltages seperately.
    n00_V_guess  = nV0_list[0]
    
    ### positive bias
    n_list_VP,convglist_VP =  self_consistent_trapz_mixing_in(ef,V_list_pos_bias,
                              n00_V_guess,
                              max_iteration,energies,
                                tol,tol_nintegrand,
                                alpha,
                            ndensity_energy,*args,
                            plotbool=plotbool,printbool=printbool)



    ### negative bias
    n_list_VMprime,convglist_VMprime =  self_consistent_trapz_mixing_in(ef,V_list_neg_bias,
                              n00_V_guess,
                              max_iteration,energies,
                                tol,tol_nintegrand,
                                alpha,
                            ndensity_energy,*args,
                            plotbool=plotbool,printbool=printbool)


    n_list_VM = np.array([n_list_VMprime[-1-i] for i in range(len(V_list_neg_bias))])
    convglist_VM = np.array([ convglist_VMprime[-1-i] for i in range(len(V_list_neg_bias)) ] )

    #Join all densities,booleans and voltages into one list.
    n_list_total = np.concatenate((n_list_VM,np.concatenate((nV0_list,n_list_VP)) ) )
    convglist_total = np.concatenate((convglist_VM,np.concatenate((convglistV0,convglist_VP))))
    
    
    
    return n_list_total,convglist_total


##############
#### Determine Fermi Level where system is half-filled

def self_consistent_trapz_mixing_in_fermi_energy(eflist,
                                  n00_list_guess,
                                  max_iteration,energies,
                                    tol,tol_nintegrand,
                                    alpha,
                                ndensity_energy,*args,plotbool=False,printbool=False):
    
    '''
    returns electron densities after self-consistent calculation with list of booleans if densities have converged.
    
    Input
    
    ef = Fermi level
    V_list = list of bias voltages
    n00_list_guess = intial guess for first in V_list
    max_iteration = maximum number of iterations 
    energies = energies to integrate over 
    tol = tolerance on electron densities
    tol_nintegrand = tolerance of integrand of electron densities
    alpha = mixing paramter [0,1]
    ndensity_energy = electron densities as function of energy
    *args = arguments of "ndensity_energy"
    plotbool = boolean if True plots electron density as function of iterations for every bias voltage
    
    Ouput:
    n_list    = list of electron densities as funciton of bias voltage
    convglist = list of booleans indicating if densities converged (True) or dit not converge(False)
    
    '''
    
    n_list = []
    convglist = []
  
    
    for i in range(len(eflist)):
        ef = eflist[i]
        muL,muR = ef, ef
        
        
        
        if i ==0:
            n00_list_init = n00_list_guess
            

              

        if i !=0:
                
            n00_list_init = n_list[i-1] #initial guess.
            
        
        if plotbool== True and printbool==True:
            print ('--- ef = {} ---'.format(ef))
            nlist_k,zero_bool = iteration_linear_mixing_plot(n00_list_init,
                                                             muL, muR,
                                                           max_iteration ,energies,
                                                             tol,tol_nintegrand,
                                                             alpha,
                                                            ndensity_energy,*args
                                   )
        if plotbool== False and printbool == True:
            print ('--- ef = {} ---'.format(ef))
            nlist_k,zero_bool = iteration_linear_mixing(n00_list_init,
                                                             muL, muR,
                                                           max_iteration ,energies,
                                                             tol,tol_nintegrand,
                                                             alpha,
                                                            ndensity_energy,*args
                                   )
            
        if printbool==False:
            nlist_k,zero_bool = iteration_linear_mixing_noprint(n00_list_init,
                                                             muL, muR,
                                                           max_iteration ,energies,
                                                             tol,tol_nintegrand,
                                                             alpha,
                                                            ndensity_energy,*args
                                   )
        
        n_list.append(nlist_k)
        convglist.append(zero_bool)
        
        
    return n_list,convglist



