# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.stdio cimport printf, fflush, stdout
from libc.stdlib cimport abort, malloc, free
from libc.math cimport floor, sqrt, round, abs, exp, acos, asin, cos, sin, log
from libc cimport bool as cbool
from cython cimport boundscheck, wraparound, numeric, floating, integral, cdivision, inline
import numpy as np
cimport numpy as cnp
from cython.parallel cimport parallel, prange
import ctypes
cimport openmp
import multiprocessing

from numpy.ctypeslib import ndpointer

"define a pointer for 1D arrays"
_doublep  = ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
"define a pointer for 1D arrays INT "
_intp  = ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')
"define a pointer for 2D arrays"
_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C')
"function to convert 2D array into a proper format for C"
def _c_2d_inp(x):
    return (x.__array_interface__['data'][0]
            + np.arange(x.shape[0]) * x.strides[0]).astype(np.uintp)

cdef double** c_2d_inp(double[:,:] array) nogil:
    
    cdef double **row_pointers = <double **>malloc(array.shape[0] * sizeof(double *))
    for i in range(array.shape[0]):
        row_pointers[i] = &array[i, 0]
    return row_pointers



cdef extern from "B02_rsd.h":
    cdef void ext_bk_mp(double **tr, double **tr2, double **tr3, double **tr4, double *log_km, double *log_pkm, double *cosm_par, double redshift, int fit_full,int kp_dim,  
int num_tr, int num_tr2,int num_tr3,int num_tr4, double *bk_mipj_ar) nogil
    


cpdef bk_multip(double[:,:] tr, double[:,:] tr2, double[:,:] tr3, double[:,:] tr4, double[:] kp, double[:] pk, double[:] cosm_par, double redshift,fit_full=1):
    bk_out = np.zeros(tr.shape[0]+tr2.shape[0]+tr3.shape[0]+tr4.shape[0], dtype=np.double)
    cdef double[:] bk_out_view = bk_out
    cdef double[:] logk = np.log10(kp)
    cdef double[:] logp = np.log10(pk)
    cdef double **trp = c_2d_inp(tr)
    cdef double **tr2p = c_2d_inp(tr2)
    cdef double **tr3p = c_2d_inp(tr3)
    cdef double **tr4p = c_2d_inp(tr4)
    ext_bk_mp(trp, tr2p, tr3p, tr4p, &logk[0], &logp[0], 
                   &cosm_par[0], redshift, fit_full, kp.size, tr.shape[0],tr2.shape[0],
                   tr3.shape[0],tr4.shape[0], &bk_out_view[0])
    free(trp)
    free(tr2p)
    free(tr3p)
    free(tr4p)
    
    ind,ind2,ind3 = tr.shape[0],tr2.shape[0],tr3.shape[0]

    bk0   = bk_out[:ind]
    bk200 = bk_out[ind:(ind+ind2)]
    bk020 = bk_out[(ind+ind2):(ind+ind2+ind3)]
    bk002 = bk_out[(ind+ind2+ind3):]

        
    return  bk0, bk200, bk020, bk002




