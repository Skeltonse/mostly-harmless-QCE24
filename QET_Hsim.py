"""
Computes angles and simulates a QSP/QET sequence for a given Hamiltonian
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time 

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')
#matplotlib.verbose.level = 'debug-annoying'


'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
import parameter_finder as pf
import solvers.Wilson_method as wm
from simulators.projector_calcs import BUILD_PLIST, Ep_CALL, Ep_PLOT, SU2_CHECK, UNIFY_PLIST
from simulators.qet_sim import COMPLEX_QET_SIM, COMPLEX_QET_SIM2, QET_MMNT
from scipy.linalg import expm
import simulators.matrix_fcns as mf
from HS_benchmark import HS_FCN_CHECK, get_coeffs

'''SPECIFIED BY THE USER'''
epsi=float(10**(-14))
t=20
ifsave=False
device='mac'
pathname="QET_Hsim.py"
H=mf.random_hermitian_matrix(8, 0.7)
device='mac'

'''DEFINE UNIVERSAL VARIABLES'''
filename="hsim_coeffs_epsi_" + "1.0e-14" + "_t_" + str(t) 
inst_tol=10**(-14)

'''DEFINE THE FIGURE AND DOMAIN'''
pts=50
data=np.linspace(-np.pi,np.pi,pts)
xdata=np.cos(data)
defconv='ad'


'''DEFINE PATHS FOR FILES'''
if device=='mac':
    current_path = os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"benchmark_data/")
else:
    current_path = os.path.abspath(__file__)
    coeff_path=current_path.replace(pathname, "")
    save_path=os.path.join(coeff_path,"\benchmark_data")

'''GENERATE AND RUN INITIAL CHECKS ON COEFF LISTS'''
czlist, szlist, n=get_coeffs(filename)
HS_FCN_CHECK(czlist, szlist, n, t, data, xdata, subnorm=2, plots=True)


'''BEGIN FINDING THE SOLUTION: BUILD $\mathcal{F}(z)$ and solve for $c(z), d(z)$'''
Plist, Qlist, E0, a, b, c, d, tDict=pf.PARAMETER_FIND(czlist, -1j*szlist, n, data,epsi=inst_tol, tDict={'tau':t})

def PLOT_QETSIM_WITH_GIVEN_H(Plist, Qlist, E0, H, ifsave=False, ):
    '''COMPUTE THE QSP ORACLE'''
    U, evals, evecs, Hevals=mf.UNITARY_BUILD(H, return_evals=True)
    evals_theta=np.angle(evals)
    
    '''RUN THE QSP CIRCUIT'''
    UPhi=COMPLEX_QET_SIM(U, Plist, Qlist, E0)
    '''EXTRACT THE CORRECT EIGENVALUES'''
    QETl=[]
    FCNl=[]
    for l in range(0, len(H)):
        QETl.append(QET_MMNT(UPhi,  evecs[:, l][:, np.newaxis])[0, 0])
        FCNl.append(np.exp(1j*Hevals[l]*t)/2)
    fig, axes = plt.subplots(1, figsize=(12, 6))
    QET=np.array(QETl)
    FCN=np.array(FCNl)

    indices=np.argsort(Hevals)
    Hevals=Hevals[indices]
    FCN=FCN[indices]
    QET=QET[indices]
    axes.scatter(Hevals, np.imag(QET),color='blue', marker='1',label=r'$\bra{\lambda}U_{\Phi}\ket{\lambda}_{imag}$')
    axes.plot(Hevals, np.imag(FCN), color='orange', marker='.', label=r'$\mathcal{A}_{imag}$')
    axes.scatter(Hevals, np.real(QET), color='green',marker='1',label=r'$\bra{\lambda}U_{\Phi}\ket{\lambda}_{real}$')
    axes.plot(Hevals, np.real(FCN),  color='red', marker=".", label=r'$\mathcal{A}_{real}$')
    #plt.legend()
    plt.title('QET simulation')
    
    if ifsave==True:
        plt.savefig(save_path+"QET_HS_for_t="+str(t)+".png")
    else:
        plt.show()
    return


PLOT_QETSIM_WITH_GIVEN_H(Plist, Qlist, E0, H, ifsave=True, )


