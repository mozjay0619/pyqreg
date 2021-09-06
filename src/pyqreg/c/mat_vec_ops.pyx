from cython cimport boundscheck, wraparound, cdivision, nonecheck, nogil

import numpy as np
cimport numpy as np
np.import_array()


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef void mv_mul(double* A, 
                 double* X,
                 double* C,
                 int N,
                 int D,
                 int transposeX):
    
    cdef int i, j
    
    if transposeX==0:
        for j in range(D):
            for i in range(N):
                C[i + N*j] = A[i + N*j] * X[j]
                
    else:
        for j in range(D):
            for i in range(N):
                C[i + N*j] = A[i + N*j] * X[i]
        
@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double* negative_assign(double* a, double* b, int n):
    
    cdef int i
    
    for i in range(n):
        b[i] = -a[i]
        
    return &b[0]

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void vs_mul(double* a, double b, double* c, int n):
    
    cdef int i
    
    for i in range(n):
        c[i] = a[i]*b

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double* vv_sub(double* a, double* b, double* c, int n):
    
    cdef int i
    
    for i in range(n):
        c[i] = a[i] - b[i]
        
    return &c[0]
        
@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double* vv_mul(double* a, double* b, double* c, int n):
    
    cdef int i
    
    for i in range(n):
        c[i] = a[i]*b[i]
        
    return &c[0]

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double* greater_than_0(double* a, double* b, int n):
    
    cdef int i
    
    for i in range(n):
        
        if a[i] > 0:
            b[i] = 1
            
        else:
            b[i] = 0
            
    return &b[0]
    
@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double* equals_0(double* a, double* b, int n):
    
    cdef int i
    
    for i in range(n):
        
        if a[i] == 0:
            b[i] = 1
            
        else:
            b[i] = 0
            
    return &b[0]

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef void row_add(double* X, double* Y, int p):
    
    cdef int i
    
    for i in range(p):
        
        X[i] += Y[i]
