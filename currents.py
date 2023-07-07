#!/usr/bin/env python
# coding: utf-8

#Author: Karssien Hero Huisman


import numpy as np 
from matplotlib import pyplot as plt



def calc_bound_integrand_current(elist,nislist,muL,muR,nround,
                                 integrand_current,*args):
    
    '''
    Input:
    elist = guess where upper and lower bound of integrand will tend to zero.
    nround = numerical order one wants to neglect
    f = considered function
    args = arguments of that function
    Output:
    lower and upper bound [elower,eupper] outside which the integrand is to the numerical order given by nround.
    '''
    
    
    e_lower = min(elist) - 1
    
    
    
    integrand_lower = abs( np.round( integrand_current(np.array([e_lower]),
                                                       nislist,muL,muR,
                                 *args ),nround
                                   )
                         )[0] 
    
    
    
    
    while integrand_lower != 0:
        e_lower -= 0.1
        integrand_lower = abs( np.round( integrand_current(np.array([e_lower]),
                                                       nislist,muL,muR,
                                 *args ),nround
                                   )
                         )[0] 
        
        
        
        
    e_upper = max(elist) + 1
    
    integrand_upper  = abs( np.round( integrand_current(np.array([e_upper]),
                                                       nislist,muL,muR,
                                 *args ),nround
                                   )
                         )[0]
    
    while integrand_upper != 0:
        e_upper += 0.1
        integrand_upper  = abs( np.round( integrand_current(np.array([e_upper]),
                                                       nislist,muL,muR,
                                 *args ),nround
                                   )
                         )[0]
    
    
    
    
    return e_lower,e_upper


def trapz_integrate(xarray,nislist,muL,muR,integrand_current,
                                    *args):
    
    '''
    Returns integral over interval with trapezium method.
    
    --------------
    Input:
    xarray = array of values to integrat over
    y = function
    *args = arguments of function y
    
    Returns
    integral of function y with arguments *args over the interval xarray
    
    '''
    yarray = integrand_current(xarray,nislist,muL,muR,*args)
    
    return np.trapz(yarray,xarray),yarray



def current_voltage(Vlist,nV_tot,ef,
                      npoints_integrate,nround,
                     integrand_current,
                                    *args,eta=1/2,plot_bool=False):
    
    Icurrent_array = np.zeros((len(Vlist),))
    for i in range(len(Vlist)):
        Vbias = Vlist[i]
        muL,muR = ef + Vbias*eta ,ef - Vbias*(1-eta)
        nislist = nV_tot[i]
        
        elist = [muL,muR]
        e_lower,e_upper = calc_bound_integrand_current(elist,nislist,muL,muR,
                                                       nround,
                                   integrand_current,
                                    *args)


        energies_integrate = np.linspace(e_lower,e_upper,npoints_integrate)


        current,integrand = trapz_integrate(energies_integrate,
                                            nislist,muL,muR,
                                   integrand_current,
                                    *args)
        
        Icurrent_array[i] = current

        if plot_bool == True:
            plt.plot(energies_integrate,integrand)
            plt.show()
            
    return Icurrent_array



def func_bias_voltages(Vmax,Vnpoints):
    Vlist = np.linspace(-Vmax,Vmax,Vnpoints)
    boolzero = 0.0 in Vlist
    
        
    if boolzero == False:
        
        'Vbias = 0 must be in Bias voltages'
        Vlist = np.linspace(-Vmax,Vmax,Vnpoints+1)
    
        return Vlist
    
    if boolzero == True:
    
        return Vlist
    
    
    
def func_PC_list(y1list,y2list,xlist,onsager_bool=False):
    '''Input
    y1list,y2list: lists that are a function of the parameter x in xlist
    Output
    plist = list with values: 'P = (y2-y1)/(y1 + y2)' 
    xprime_list = New x parameters. Values of x for which y1(x) + y2(x) =0 are removed (0/0 is numerically impossible)'''
    
    p_list = []
    xprime_list= []
    for i in range(len(xlist)):
        x = xlist[i]
        
        y1 = y1list[i]
        y2 = y2list[i]
        
        
        
        if x!=0:
            p_list.append(100*np.subtract(y1,y2)/(y2 + y1))
            
            xprime_list.append(x)
        if x==0 and onsager_bool == True:
            p_list.append(0)
            
            xprime_list.append(x)
            
        if x==0 and onsager_bool == False:
            print('check that Onsager reciprocity holds')
            
    
    return xprime_list,p_list



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