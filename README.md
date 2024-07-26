# mostly-harmless-QCE24
QSP parameter-finding package used in "mostly harmless methods for QSP-processing or Laurent polynomials".
The paper will be presented at IEEE Quantum Week 2024 <!--, the ArXiV pre-print is found at (FILL IN)-->.
<!--This repository in unmaintained, see (FILL IN) for an actively maintained QSP parameter-finding package with more options.-->

# core functionality
This package solves the completion step of QSP-processing using the Wilson method for the Fejer probelm in complex analysis, with options to solve the decomposition step and run a QSP simulation. Coefficent lists of common bounded-error QSP functions is also given, although not all coefficient lists can be eaisly generated in floating-point precision.

# available demos
QSP-parameters can be generated for Hamitlonian simulation (HS) or for implementing a Fourier series with randomly generated coefficients (random) using:
HS_benchmark.py
random_benchmark.py

One can also run a QSP simulation implementing a polynomial in $e^{i\theta}$ for any given Hamiltonian with eigenvalues $\lambda=\cos\theta$. This is
QSP_demo.py
where a user must specify the following (default is Hamiltonian simulation with a random Hamiltonian):
H: the Hamiltonian as a np.array
n: degree of the polynomial
epsi: the desired precision of the solution
ccoeff: a 2n+1 length np array with the coefficients of a reciprocal, real-on-circle polynomial
scoeff: a 2n+1 length np array with the coefficients of an anti-reciprocal, real-on-circle polynomial

# available functions
The coefficnet lists of polynomails for Hamiltonian Simulation and random Fourier series can be generated
Jacobi-Angler expansion: functions/Hsim.jl
Random polynomials: functions/random.py

# unsuccessful polynomials
Some common QSP and QSVT polynomials are difficult to generate in the correct basis or our method. 
Mathematica code demonstrating this difficulty is given for the following polynomials:
rectangule function: functions/fl16_notes
Normalized inverse function: functions/fgslw19_notes.nb
Threshold minmax polynomial: functions/flt_20_precision_tests.nb
