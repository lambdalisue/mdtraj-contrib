import numpy as np
cimport numpy as np
import scipy.optimize as optimize

ctypedef np.ndarray ARRAY
ctypedef np.int_t INT_t
ctypedef np.float64_t DOUBLE_t

# Ref: http://phe.rockefeller.edu/LogletLab/whitepaper/node17.html
def bootstrap_fit(
        object fn,
        ARRAY[DOUBLE_t, ndim=1] xdata,
        ARRAY[DOUBLE_t, ndim=1] ydata,
        ARRAY[DOUBLE_t, ndim=1] p0,
        float yerr_systematic=0.0, float nsigma=1., int nrandom=100):
    cdef object errorfn = lambda p, x, y: y - fn(x, *p)
    cdef ARRAY[DOUBLE_t, ndim=1] params = optimize.leastsq(
        errorfn, p0, args=(xdata, ydata))[0]
    # Get the standard deviation of the residuals
    cdef ARRAY[DOUBLE_t, ndim=1] residuals = errorfn(params, xdata, ydata)
    cdef float stdev_res = np.std(residuals)
    cdef float stdev_err = np.sqrt(stdev_res**2 + yerr_systematic**2)
    cdef list params_list = []
    cdef int i
    cdef ARRAY[DOUBLE_t, ndim=1] rdelta
    cdef ARRAY[DOUBLE_t, ndim=1] rydata
    cdef int n = len(ydata)
    for i in range(nrandom):
        rdelta = np.random.normal(0, stdev_err, n)
        rydata = ydata + rdelta
        params = optimize.leastsq(errorfn, p0, args=(xdata, rydata))[0]
        params_list.append(params)
    cdef ARRAY[DOUBLE_t, ndim=2] psl = np.array(params_list)
    cdef ARRAY[DOUBLE_t, ndim=1] popts = np.mean(psl, 0)
    cdef ARRAY[DOUBLE_t, ndim=1] perrs = nsigma * np.std(psl, 0)
    return popts, perrs
