"""
An algorithm for computing the inverse normal cumulative distribution function
devised by Peter John Acklam.
https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/#C

Written in C by V. Natarajan (Kanchipuram (near Madras), India)
https://web.archive.org/web/20151030215621/http://home.online.no/~pjacklam/notes/invnorm/impl/natarajan/
"""

from libc.math cimport erfc, sqrt, log, pi, exp

from cython cimport boundscheck, wraparound, cdivision, nonecheck, nogil

import numpy as np
cimport numpy as np
np.import_array()


DTYPE = np.float64


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double _invnormal(double p):
    """Compute the inverse normal cumulative distribution function.

    Translated from C code written by V. Natarajan.

    Parameters
    ----------
    p : double

    """
    cdef double A1 = -3.969683028665376e+01
    cdef double A2 =  2.209460984245205e+02
    cdef double A3 = -2.759285104469687e+02
    cdef double A4 =  1.383577518672690e+02
    cdef double A5 = -3.066479806614716e+01
    cdef double A6 =  2.506628277459239e+00

    cdef double B1 = -5.447609879822406e+01
    cdef double B2 =  1.615858368580409e+02
    cdef double B3 = -1.556989798598866e+02
    cdef double B4 =  6.680131188771972e+01
    cdef double B5 = -1.328068155288572e+01

    cdef double C1 = -7.784894002430293e-03
    cdef double C2 = -3.223964580411365e-01
    cdef double C3 = -2.400758277161838e+00
    cdef double C4 = -2.549732539343734e+00
    cdef double C5 =  4.374664141464968e+00
    cdef double C6 =  2.938163982698783e+00

    cdef double D1 =  7.784695709041462e-03
    cdef double D2 =  3.224671290700398e-01
    cdef double D3 =  2.445134137142996e+00
    cdef double D4 =  3.754408661907416e+00
    
    cdef double P_LOW = 0.02425
    cdef double P_HIGH = 0.97575
    
    cdef double x
    cdef double q, r, u, e
    
    if ((0.0 < p) & (p < P_LOW)):
        q = sqrt(-2.0*log(p))
        x = (((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1)
    
    else:
        
        if ((P_LOW <= p) & (p <= P_HIGH)):
            q = p - 0.5
            r = q*q
            x = (((((A1*r+A2)*r+A3)*r+A4)*r+A5)*r+A6)*q /(((((B1*r+B2)*r+B3)*r+B4)*r+B5)*r+1.0)
        
        else:
            
            if ((P_HIGH < p) & (p < 1.0)):
                q = sqrt(-2.0*log(1.0-p))
                x = -(((((C1*q+C2)*q+C3)*q+C4)*q+C5)*q+C6) / ((((D1*q+D2)*q+D3)*q+D4)*q+1.0)
                
    if((0 < p) & (p < 1)):
        
        e = 0.5 * erfc(-x/sqrt(2.0)) - p
        u = e * sqrt(2.0*pi) * exp(x*x/2.0)
        x = x - u/(1.0 + x*u/2.0)

    return x

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def invnormal(double p):

    return _invnormal(p)

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef double _normalden(double z):
    """Compute the standard normal density function.

    Parameters
    ----------
    z : double

    """
    return exp(-z**2.0 / 2.0) / (sqrt(pi * 2.0))

@boundscheck(False)
@wraparound(False)
@cdivision(True)
def normalden(double z):
    
    return _normalden(z)

    

