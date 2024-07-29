# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:04:20 2023

@author: Shawn Skelton
Script with a few definitions of randomly generated pairs of Laurent polynomials
"""

import numpy as np
import functions.laur_poly_fcns as lpf 
import parameter_finder as pf

def RAND_CHEBY_GENERATOR(n):
    """
    Randomly generates n+1, n coefficients for each polynmial list.
    resulting Lauent polynomials are real-on-circle and reciprocal by definition
    resulting Lauent polynomials are real-on-circle and reciprocal/anti-reciprocal by definition
    however constituents a, b are not even or odd wrt Chebyshev basis

    input:
    n: float, max degree of desired polynomial
    """
    if n%2==0:
        clist=np.random.rand(n+1)
        slist=np.append(np.append(0, np.random.rand(n-1)), 0)
    elif n%2==1:
        clist=np.append(np.random.rand(n), 0)
        slist=np.append(0, np.random.rand(n))
    return clist, slist


def RAND_LIMIT_CHEBY_GENERATOR(n, nz=5):
    """
    Randomly generates n+1, n coefficients for each polynmial list, now restricts the number of non-zero coefficients.
    The highest five non-zero coefficients are randomly selected for each of a, b and the rest are set to zero
    resulting Lauent polynomials are real-on-circle and reciprocal/anti-reciprocal by definition
    however constituents a, b are not even or odd wrt Chebyshev basis

    input:
    n: float, max degree of desired polynomial
    nz: the number of non-zero coefficients
    """
    ####need to double check parity here###
    if n%2==0:
        clist=np.append(np.zeros(n-nz+1), np.random.rand(nz))
        slist=np.append(np.append(np.zeros(n-nz), np.random.rand(nz) ),0 )
        
    elif n%2==1:
        clist=np.append(np.append(np.zeros(n-nz), np.random.rand(nz)), 0)
        slist=np.append(np.append(0, np.random.rand(n-nz)), np.random.rand(nz))
        
    return clist, slist

def RAND_JUMBLE_CHEBY_GENERATOR(n, nz):
    """
    Randomly generates coefficient lists for Fourier series, restricts the number of non-zero coefficients.
    The highest five non-zero coefficients are randomly selected for each of a, b and the rest are set to zero
    
    resulting Lauent polynomials are real-on-circle and reciprocal/anti-reciprocal by definition

    input:
    n: float, max degree of desired polynomial
    nz: the number of non-zero coefficients
    """
    ###DEFINE THE TWO COEFFICIENT LISTS WITH LEADING NON-ZERO TERMS###
    if n%2==0:
        clist=np.append(np.zeros(n), np.random.rand(1))
        slist=np.append(np.append(np.zeros(n-1), np.random.rand(1)), 0)
    elif n%2==1:
        clist=np.append(np.append(np.zeros(n-1), np.random.rand(1)), 0)
        slist=np.append(np.zeros(n), np.random.rand(1))

    ###NOW ASSIGN RANDOMLY POSITIONED NON-ZERO COEFFICIENTS###
    ##iterate over the number of non-zero coefficients in both lists##
    for i in range(0, 2*nz):
        index=np.random.randint(0,high=n)
        if index%2==0:
            clist[index]=np.random.rand(1)
        else:
            slist[index]=np.random.rand(1)

    return clist, slist

def GET_NORMED(clist, slist,n,  subnorm=4):
    """
    Computes the Laurent coefficient lists of subnormalized clsit, slist

    inputs:
    clist, slist: n+1 length arrays, coeffciients of the even and od Fourier series
    n: float, the max degree of the Laurent series
    subnorm: flaot, option to adjust the subnormalization. Default is $|f|\leq 0.5$

    returns:
    clist, slist n+1 np arrays subnormalized
    czlist, szlist 2n+1 np arrays, subnormalized Laurent coefficient lists
    epsi: float, the max error between the polynomial build from czlist or szlist and a completely real-on-circle polynomial
    """
    czlist, szlist=lpf.GET_LAURENT(clist, slist, n)
    
    valsa, epsia=lpf.REAL_CHECK(czlist, n, tol=10**(-14), fcnname='czlist', giveprecision=True)
    valsb, epsib=lpf.REAL_CHECK(szlist, n, tol=10**(-14), fcnname='szlist', giveprecision=True)
    maxvala=max(abs(valsa))
    maxvalb=max(abs(valsb))
    
    prefactora=1/maxvala/(subnorm)
    prefactorb=1/maxvalb/(subnorm)
    

    return prefactora*clist, prefactorb*slist, prefactora*czlist, prefactorb*szlist, epsia+epsib
