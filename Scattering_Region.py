
#Author: Karssien Hero Huisman

import numpy as np
from matplotlib import pyplot as plt
import Geometry_Helix

sigmaz = np.diag(np.array([1,-1],dtype = complex))


def chi(R,B,t):
    Rt = np.dot(R,t)
    RtB = np.dot(R,np.cross(t,B))
    return 0.5*Rt*RtB


def phi_mn(B,tvec,Phi0,a,c,M,N,chirality,offset =1):

    
    mlist = []
    phi_mn_list = []
    for n in range(1,M*N  + 1  -offset):
        Rn = Geometry_Helix.rvec(n,a,c,M,N,chirality)
        Rm = Geometry_Helix.rvec(n + offset,a,c,M,N,chirality)


        phi_mn = (2*np.pi/Phi0)*(0.5*np.dot(B,np.cross(Rm,Rn)) + chi(Rn,B,tvec) - chi(Rm,B,tvec))
        phi_mn_list.append(phi_mn)
        
        mlist.append(n)
        
    return np.array(phi_mn_list)



def Hamiltonian_helical_chain(t,tprime,
                                              B,tvec,Phi0,a,c,M,N,chirality,
                                              plot_bool=False):
    
    dim = int(M*N)
    H0 = np.zeros((dim,dim),dtype =complex)

    
    ### NN hopping
    list_hop = -t*np.exp(1j*phi_mn(B,tvec,Phi0,a,c,M,N,chirality,offset = 1))
    H_hop_forward = np.diag(list_hop,1)
    H_hop_NN = H_hop_forward + np.conjugate(np.transpose(H_hop_forward))
    
    
    ### NNN hopping
    list_hop_NNN = -tprime*np.exp(1j*phi_mn(B,tvec,Phi0,a,c,M,N,chirality,offset = 2))
    H_hop_forward_NNN = np.diag(list_hop_NNN,2)
    H_hop_NNN = H_hop_forward_NNN + np.conjugate(np.transpose(H_hop_forward_NNN))

    ### Onsite Hamiltonian
    H0    = H_hop_NN + H_hop_NNN
    H0_spin = np.kron(H0,np.identity(2))
    if plot_bool == True:
        plt.title('Hamiltonian : real part')
        plt.imshow(H0_spin.real)
        plt.colorbar()
        plt.show()
        
        
        plt.title('Hamiltonian : imaginary part')
        plt.imshow(H0_spin.imag)
        plt.colorbar()
        plt.show()
    return H0_spin


def coupling_matrix(dim,gammaL,gammaR,pz,plot_bool=False):
    GammaL = np.zeros((dim,dim),dtype=complex)
    GammaR = np.zeros((dim,dim),dtype=complex)
    
    GammaL[0,0]   = 1
    GammaL[1,1]   = 1
    GammaR[-2,-2] = 1
    GammaR[-1,-1] = 1
    
    GammaL_spin = np.kron(GammaL,np.identity(2))
    GammaR_spin = np.kron(GammaR,np.identity(2))
    if plot_bool == True:
        plt.title('GammaL : real part')
        plt.imshow(GammaL_spin.real)
        plt.colorbar()
        plt.show()
        
        plt.title('GammaR : real part')
        plt.imshow(GammaR_spin.real)
        plt.colorbar()
        plt.show()
        
        
    GammaL_spin[0:2,0:2] = gammaL*(np.identity(2) + pz*sigmaz)
    GammaL_spin[2:4,2:4] = gammaL*(np.identity(2) + pz*sigmaz)
    
    return GammaL_spin,GammaR_spin

def system(t,tprime,
           Bvec,tvec,Phi0,
           a,c,M,N,chirality,
                   gammaR,gammaL,pz,
          plot_bool=False):
    H0_spin = Hamiltonian_helical_chain(t,tprime,
                                                  Bvec,tvec,Phi0,a,c,M,N,chirality,
                                                  plot_bool)

    dim = int(H0_spin.shape[0]/2)
    GammaLP_spin,GammaR_spin = coupling_matrix(dim,gammaL,gammaR,pz,
                                               plot_bool)
    
    return H0_spin,GammaLP_spin,GammaR_spin


def system_magnetizations(t,tprime,
           Bvec,tvec,Phi0,
           a,c,M,N,chirality,
                   gammaR,gammaL,pz,
          plot_bool=False):
    H0P,GammaLP,GammaR = system(t,tprime,
                                           Bvec,tvec,Phi0,
                                           a,c,M,N,chirality,
                                                   gammaR,gammaL, abs(pz),
                                          plot_bool=plot_bool
                                          )


    H0M,GammaLM,GammaR = system(t,tprime,
                                           -Bvec,tvec,Phi0,
                                           a,c,M,N,chirality,
                                                   gammaR,gammaL,-abs(pz),
                                          plot_bool=plot_bool
                                          )
    dim = H0P.shape[0]
    return  H0P,GammaLP,GammaR,H0M,GammaLM,GammaR,dim     



    
    
###########################################################################
######################### Chiral Chain With SPIN ORBIT COUPLING
###########################################################################
    
    
import kwant
import kwant.qsymm
import tinyarray

# define Pauli-matrices for convenience
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])
a=1

# Import Geometrical Paramters for construction of SOC Hamiltonian.


import kwant
import kwant.qsymm
import tinyarray

# define Pauli-matrices for convenience
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])
a=1

# Import Geometrical Paramters for construction of SOC Hamiltonian.


def make_system_straight_U0(Lm,epsilon):
    '''
    Input = Lenght of molecule
    epsilon = onsite energy
    t = NN hopping paramter (spin-independent)
    Output:
    - Kwant system of chain with spin depedent hopping
    '''
    
    a   = 1
    lat =  kwant.lattice.square(a,norbs = 2) # lattice with 2 spin degree of freedom
    syst = kwant.Builder()
    
    
    
     ### DEFINE LATTICE HAMILTONIAN ###
    for i in range( 1,Lm +1  ):
            syst[lat(i, 0)] =  epsilon*sigma_0 
            

   
    syst[kwant.builder.HoppingKind((1,0), lat, lat)] =  sigma_0
    syst[kwant.builder.HoppingKind((2,0), lat, lat)] =  sigma_0
    
    def hopping_1site(site1, site2,t,B,Phi0,chirality,a,c,M,N,tvec):
        '''
        Input:
        - N = number of sites per lap
        - M = number of laps
        - c = length of the molecule
        - a = radius
        - m = site label
        - chirality = boolean indicating the chirality of the helix.
        Ouput:
        - Hopping matrices (in spin space) between site1,site2 for chirality = True
        '''
        m,m2 = site1.pos
        n,j2 = site2.pos
        
        Rm = Geometry_Helix.rvec(m,a,c,M,N,chirality)
        Rn = Geometry_Helix.rvec(n,a,c,M,N,chirality)
        phi_mn = (2*np.pi/Phi0)*(0.5*np.dot(B,np.cross(Rm,Rn)) + chi(Rn,B,tvec) - chi(Rm,B,tvec))
#         print(n,m)
#         print(phi_mn)


        return sigma_0*(-t)*np.exp(phi_mn*1j)
    
    
    
    
    def hopping_2site(site1, site2,tprime,lambda1,B,Phi0,chirality,a,c,M,N,tvec):
        '''
        Input:
        - N = number of sites per lap
        - M = number of laps
        - c = length of the molecule
        - a = radius
        - m = site label
        - chirality = boolean indicating the chirality of the helix.
        Ouput:
        - Hopping matrices (in spin space) between site1,site2 for chirality = True
        '''
        m,m2 = site1.pos
        n,j2 = site2.pos
        Rm = Geometry_Helix.rvec(m,a,c,M,N,chirality)
        Rn = Geometry_Helix.rvec(n,a,c,M,N,chirality)
        phi_mn = (2*np.pi/Phi0)*(0.5*np.dot(B,np.cross(Rm,Rn)) + chi(Rn,B,tvec) - chi(Rm,B,tvec))
#         print(n,m)
#         print(phi_mn)
        
        vvector = Geometry_Helix.Vmvec(m=m,stilde=1,a=a,c=c,M=M,N=N,chirality=chirality)
        sigma_vec = np.array([sigma_x,sigma_y,sigma_z])
        
        innerproduct = 1j*lambda1*np.tensordot(vvector,sigma_vec,axes =1)

        return (-sigma_0*tprime + innerproduct)*np.exp(phi_mn*1j)
    
    
    syst[kwant.builder.HoppingKind((1,0), lat, lat)] =  hopping_1site
    syst[kwant.builder.HoppingKind((2,0), lat, lat)] =  hopping_2site

    return syst




def system_hamiltonian0(Lm,epsilon,t, tprime,lambda1,B,Phi0,chirality,a,c,M,N,tvec,
                        kmax,gammaL,gammaR,pz):
    
    '''Input
    system paramters
    Output:
    - Hamiltonian without Coulomb Interactions
    - Gamma_R,Gamma_L coupling matrices. Left lead is magnetized.
    '''

         

    system =  make_system_straight_U0(Lm,epsilon)
    kwant_sytem = kwant.qsymm.builder_to_model(system, params={'t':t,
                                                               'tprime':tprime,
                                                               'lambda1': lambda1,
                                                               'B':B,
                                                               'Phi0':Phi0,
                                                               'chirality': chirality,
                                                               'a':a,
                                                               'c':c, 
                                                               'M':M,
                                                               'N':N,
                                                              'tvec':tvec}) 


    Hamiltonian = np.array(kwant_sytem[1])
    dim = Hamiltonian.shape[0]


    #Diagonal WBL Gamma's
    GammaR = np.zeros((dim,dim))
    GammaLP = np.zeros((dim,dim))


    for i in range(kmax):
        if i % 2 == 0:
            GammaLP[i,i] = gammaL*(1+pz)
        if i % 2 == 1:
            GammaLP[i,i] = gammaL*(1-pz)


        GammaR[-i-1,-i-1] = gammaR


    return GammaR,GammaLP,Hamiltonian,dim

def chiral_chain_soc(epsilon,t, tprime,lambda1,B,Phi0,chirality,a,c,M,N,tvec,
                        kmax,gammaL,gammaR,pz):
    
    
    GammaR,GammaLP,H0P,dim = system_hamiltonian0(int(N*M),epsilon,
                                                         t, tprime,lambda1,
                                                         B,Phi0,chirality,a,c,M,N,tvec,
                                                        kmax,gammaL,gammaR,abs(pz))
    GammaR,GammaLM,H0M,dim = system_hamiltonian0(int(N*M),epsilon,
                                                         t, tprime,lambda1,
                                                         -B,Phi0,chirality,a,c,M,N,tvec,
                                                        kmax,gammaL,gammaR,-abs(pz))
    
    return GammaR,GammaLP,H0P,GammaR,GammaLM,H0M,dim

