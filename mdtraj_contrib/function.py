import numpy as np
import scipy.optimize
import mdtraj_contrib.optimize as optimize


def exponential_decay(t, A, l, y0):
    """y = A * exp(-t * l) + y0"""
    return A * np.exp(-t * l) + y0


def fit_exponential_decay(t, y, nsigma=1., nrandom=100):
    # Estimate an initial parameter with curve_fit
    p0 = scipy.optimize.curve_fit(exponential_decay, t, y)[0]
    # Find optmized parameter by bootstrap fit
    return optmize.bootstrap_fit(
        exponential_decay,
        t.astype(float),
        y.astype(float),
        np.array(p0),
        nsigma=nsigma,
        nrandom=nrandom,
    )
