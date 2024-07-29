"""
Used to be the main HS file, now it has a similar function to QET_test_script.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time 

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')
#matplotlib.verbose.level = 'debug-annoying'


'''LOCAL IMPORTS'''
import functions.laur_poly_fcns as lpf 
import parameter_finder as pf
from functions.matrix_inverse import CHEBY_INV_COEFF_ARRAY
import solvers.Wilson_method as wm
from simulators.projector_calcs import BUILD_PLIST, Ep_CALL, Ep_PLOT, SU2_CHECK, UNIFY_PLIST
from simulators.angle_calcs import PROJ_TO_ANGLE, Wx_TO_R, W_PLOT, PAULI_CHECK, HAAHR_PLOT, HAAHW_PLOT
from simulators.qet_sim import COMPLEX_QET_SIM, COMPLEX_QET_SIM2, QET_MMNT

from scipy.linalg import expm
import simulators.matrix_fcns as mf
from simulators.unitary_calcs import *

'''SPECIFIED BY THE USER'''
epsi=float(10**(-14))
t=20
ifsave=False

'''DEFINE UNIVERSAL VARIABLES'''
filename="hsim_coeffs_epsi_" + "1.0e-14" + "_t_" + str(t) 
inst_tol=10**(-14)
H=mf.random_hermitian_matrix(8, 0.7)

'''DEFINE THE FIGURE AND DOMAIN'''
fig, axes = plt.subplots(1, figsize=(6, 12))
pts=50
data=np.linspace(-np.pi,np.pi,pts)
xdata=np.cos(data)
defconv='ad'


'''DEFINE PATHS FOR FILES'''
pathname="QET_Hsim.py"
current_path = os.path.abspath(__file__)
coeff_path=current_path.replace(pathname, "")
save_path=os.path.join(coeff_path,"benchmark_data")

'''SHOULD ADD THE PREFACTOR AS A VARIABLE IN THE CSV FILE'''
def get_coeffs(filename):
    current_path = os.path.abspath(__file__)
    coeff_path=current_path.replace("\QET_Hsim.py", "")
    abs_path=os.path.join(coeff_path,"csv_files", filename+".csv")
    with open(abs_path, 'r') as file:
        csv_reader = csv.reader(file)
        column1=[]
        column2=[]
        column3=[]
        next(csv_reader)

        for row in csv_reader:
            col1, col2, col3= row[:3]  # Unpack the first three columns
            column1.append(col1)
            column2.append(col2)
            column3.append(col3)
    ###I keep imaginary numbers in the coeff arrays so each array will produce a real-on-circle poly without pain
    ccoeff=np.array(column1).astype(float)
    scoeff=1j*np.array(column2).astype(float)
    n=int(column3[0])
    return ccoeff, scoeff, n

czlist, szlist, n=get_coeffs(filename)
#czlist=czlist[1:2*n]/2
#szlist=szlist[1:2*n]/2

'''INITIAL CHECKS ON COEFF LISTS; SHOULD EVENTUALLY BE OPTIONAL''' 
def HS_FCN_CHECK(czlist, szlist, n, data, xdata):
    fl=lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+1j*lpf.LAUR_POLY_BUILD(-1j*szlist, n, np.exp(1j*data))
    targetfcn=np.exp(1j*xdata*t)/2
    axes[0].plot(xdata, np.real(fl), marker='.', label=r'$\mathcal{A}_{real}$')  
    axes[0].plot(xdata, np.real(targetfcn),label=r'$e^{it\lambda}_{real}$', )
    axes[0].plot(xdata, np.imag(fl), marker='.', label=r'$\mathcal{A}_{imag}$')  
    axes[0].plot(xdata, np.imag(targetfcn),label=r'$e^{it\lambda}_{imag}$', )
    axes[0].set_title("verify the approximation is alright")
    axes[0].legend()
    lpf.REAL_CHECK(czlist, n, theta=data, tol=inst_tol,fcnname='czlist')
    lpf.REAL_CHECK(-1j*szlist, n,theta=data,  tol=inst_tol, fcnname='szlist')

####FIND THE QSP CIRCUIT###
'''BEGIN FINDING THE SOLUTION: BUILD $\mathcal{F}(z)$ and solve for $c(z), d(z)$'''
Plist, Qlist, E0, a, b, c, d, tDict=pf.ANGLE_FIND(czlist, -1j*szlist, n, data)

def PLOT_QETSIM_WITH_GIVEN_H(Plist, Qlist, E0, H, ifsave=False, ):
    '''COMPUTE THE QSP ORACLE'''
    U, evals, evecs, Hevals=mf.UNITARY_BUILD(H, return_evals=True)

    '''RUN THE QSP CIRCUIT'''
    UPhi=COMPLEX_QET_SIM(U, Plist, Qlist, E0)
    '''EXTRACT THE CORRECT EIGENVALUES'''
    QETl=[]
    FCNl=[]
    for l in range(0, len(H)):
        QETl.append(QET_MMNT(UPhi,  evecs[:, l][:, np.newaxis])[0, 0])
        FCNl.append(np.exp(1j*Hevals[l]*t)/2)

    QET=np.array(QETl)
    FCN=np.array(FCNl)
    axes.scatter(Hevals, np.imag(QET),color='blue', marker='1',label=r'$\bra{\lambda}U_{\Phi}\ket{\lambda}_{imag}$')
    axes.scatter(Hevals, np.imag(FCN), color='orange', marker='.', label=r'$\mathcal{A}_{imag}$')
    axes.scatter(Hevals, np.real(QET), color='green',marker='1',label=r'$\bra{\lambda}U_{\Phi}\ket{\lambda}_{real}$')
    axes.scatter(Hevals, np.real(FCN),  color='red', marker=".", label=r'$\mathcal{A}_{real}$')


    if ifsave==True:
        plt.savefig(save_path+"QET_HS_for_t="+str(t)+".png")
    else:
        plt.show()
    return

def PLOT_REALQSPSIM_WITH_DATA(Plist, Qlist, E0, epsi, data, ifsave=False, ):
    '''COMPUTE PARAMS FOR NEW QSP CIRCUIT'''
    philist=PROJ_TO_ANGLE(Plist, E0, n, recip=defconv,tol=epsi)
    philistr=Wx_TO_R(philist)

    '''RUN THE QSP CIRCUITS'''
    Ep_PLOT(Plist, Qlist, E0, n, a, b, data,  ax=axes, )
    #W_PLOT(philist, n,  data, conv=defconv, ax=axes, )
    #HAAHR_PLOT(philistr, n,  data, conv='R', ax=axes, )

    return 

def PLOT_QSPSIM_WITH_DATA(Plist, Qlist, E0, data, a, b, ifsave=False, withcomp=False):
    Ep_PLOT(Plist, Qlist, E0, n, a, b, data,  ax=axes, )
    if withcomp==True:
        axes.scatter(data, np.real(lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+ 1j*lpf.LAUR_POLY_BUILD(-1j*szlist, n, np.exp(1j*data))))
        axes.scatter(data,np.imag(lpf.LAUR_POLY_BUILD(czlist, n, np.exp(1j*data))+ 1j*lpf.LAUR_POLY_BUILD(-1j*szlist, n, np.exp(1j*data))))

    if ifsave==True:
        plt.savefig(save_path+"REQSP_HS_for_t="+str(t)+".png")
    else:
        plt.show()
    return

PLOT_QETSIM_WITH_GIVEN_H(Plist, Qlist, E0, H, ifsave=False, )
# PLOT_REALQSPSIM_WITH_DATA(Plist, Qlist, E0, epsi, data,)
plt.show()
# def GQSP_SIM(thetalist, philist, lambd, U, convent=np.array([[1], [1]])/2):
#     clength=len(thetalist)
#     Ul=len(U)
#     SystIdent=np.identity(Ul)

#     CzU=np.kron(np.array([[1, 0], [0, 0]]), U)+np.kron(np.array([[0, 0], [0, 1]]), SystIdent)
#     Uqsp=np.kron(RU(thetalist[0], philist[0], lambd), SystIdent)
    
#     for i in range(1, clength):
#         iterU=np.kron(RU(thetalist[i], philist[i]), SystIdent)@CzU
#         Uqsp=iterU@Uqsp

    
#     projtoconvent=np.kron(convent@np.conj(convent).T, SystIdent)@Uqsp@np.kron(convent@np.conj(convent).T, SystIdent)
#     Ured=np.trace(projtoconvent.reshape(2, Ul, 2, Ul), axis1=0, axis2=2)   
#     return Ured

# def GQSP_LSIM(thetalist, philist, lambd, U, k, convent=np.array([[1], [1]])/2):
#     clength=len(thetalist)
#     Ul=len(U)
#     SystIdent=np.identity(Ul)

#     CzU=np.kron(np.array([[1, 0], [0, 0]]), U)+np.kron(np.array([[0, 0], [0, 1]]), SystIdent)
#     CzUdag=np.kron(np.array([[1, 0], [0, 0]]), U)+np.kron(np.array([[0, 0], [0, 1]]), SystIdent)
#     Uqsp=np.kron(RU(thetalist[0], philist[0], lambd), SystIdent)
    
#     for i in range(1, clength-k):
#         iterU=np.kron(RU(thetalist[i], philist[i]), SystIdent)@CzU
#         Uqsp=iterU@Uqsp

#     for i in range(clength-k, clength):
#         iterU=np.kron(RU(thetalist[i], philist[i]), SystIdent)@CzUdag
#         Uqsp=iterU@Uqsp 

#     projtoconvent=np.kron(convent@np.conj(convent).T, SystIdent)@Uqsp@np.kron(convent@np.conj(convent).T, SystIdent)
#     Ured=np.trace(projtoconvent.reshape(2, Ul, 2, Ul), axis1=0, axis2=2)   
#     return Ured

# Pcoeffs, Qcoeffs, Podd, Qeven, Gn=REINDEX(a, b, d, c)

# P_approx=lpf.LAUR_POLY_BUILD(Pcoeffs, np.int32(Gn/2), np.exp(1j*data)**2)+np.exp(1j*data)*lpf.LAUR_POLY_BUILD(Podd, np.int32(Gn/2), np.exp(1j*data)**2)#*np.exp(1j*thdata)**n
# Q_approx=np.exp(1j*data)*lpf.LAUR_POLY_BUILD(Qcoeffs, np.int32(Gn/2), np.exp(1j*data)**2) #*np.exp(1j*data)**n
# P_approx=(np.exp(1j*data)**n)*np.exp(1j*data)*lpf.LAUR_POLY_BUILD(Pcoeffs, np.int32((n-1)/2), np.exp(1j*data)**2)#*np.exp(1j*thdata)**n
# Q_approx=np.exp(1j*data)*Q_POLY_BUILD(Qcoeffs, n, np.exp(1j*data)**2) #*np.exp(1j*data)**n

# targetfcn=(np.cos(xdata*t)/2*np.exp(1j*data)+1j*np.sin(xdata*t)/2)*np.exp(1j*data)**n

targP=np.exp(1j*data)*np.cos(xdata*t)/2#*np.exp(1j*thdata)**n
targQ=1j*np.sin(xdata*t)/2 #*np.exp(1j*data)**n

# print(P_approx*np.conj(P_approx)+Q_approx*np.conj(Q_approx))
# Pgenlist=Pcoeffs+Podd
# Qgenlist=Qcoeffs+Qeven
# print(len(Pgenlist))
# thetalist,philist, lambd=COMPUTE_R(Pgenlist, Qgenlist, Gn)

# GU=GQSP_SIM(thetalist, philist, lambd, U, convent=np.array([[1], [1]])/np.sqrt(2))

# targetfcn=np.exp(1j*xdata*t)/2*np.exp(1j*data)**2
# axes[3].plot(data, np.real(P_approx), marker='.', label=r'$\mathcal{P}_{real}$')  
# axes[3].plot(data, np.real(targP),label=r'$e^{it\lambda}_{real}$', )
# axes[3].plot(data, np.imag(targQ), marker='.', label=r'$e^{it\lambda}_{imag}$',)  
# axes[3].plot(data, np.imag(Q_approx),label=r'$\mathcal{Q}_{imag}$' )
# axes[0].legend()


# GQSP=np.array(GQSPl)

#axes[3].scatter(Hevals, np.imag(GQSP),color='seagreen', marker='1',label=r'$\bra{\lambda}U_{gqsp}\ket{\lambda}_{imag}$')
#axes[3].scatter(Hevals, np.real(GQSP), color='gold',marker='1',label=r'$\bra{\lambda}U_{gqsp}\ket{\lambda}_{real}$')
# axes[3].set_title('Compare QSP result to the origional Function'+r"$|\mathcal{A}-\bra{\lambda}U_{\Phi}\ket{\lambda}|_{\max}=$"+ str((mf.NORM_CHECK(QETA, FCNA), 2)))


