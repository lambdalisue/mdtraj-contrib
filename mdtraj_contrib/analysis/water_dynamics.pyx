#******************************************************************************
#
# mdtraj-contrib
# Water dynamics analysis module
#
# Author:  lambdalisue
# License: GNU Public License v3 (NOTE: Based on MDAnalysis license)
#
# NOTE:
# Written code here are based on MDAnalysis.analysis.waterdynamics written by
# Alejandro Bernardin.
#
# A lot of modifications and aggressive refactorings were performed by
# lambdalisue to make it faster and compatible with MDTraj
#
#******************************************************************************
import numpy as np
import mdtraj as md
cimport numpy as np

ctypedef np.int_t INT_t
ctypedef np.float32_t FLOAT_t
ctypedef np.float64_t DOUBLE_t

cdef double lg2(double x):
    """
    Second Legendre polynomial
    Ref: https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    return (3 * x**2 - 1) / 2.

def build_dipole_vectors(object trajectory,
                         np.ndarray[INT_t, ndim=1] indexes_at_t,
                         int t):
    """
    Build a M x 3 matrix, M dipole vectros (x, y, z) of water molecules

    Args:
        trajectory (md.Trajectory) : A target trajectory instance
        indexes (numpy.ndarray) : M x 1 array of index of oxygen atoms of water
            molecules
        t (int) : A frame index

    Returns:
        M x 3 matrix : A collection of dipole vectors of water molecules
    """
    cdef np.ndarray[FLOAT_t, ndim=2] coords = trajectory.xyz[t]
    cdef np.ndarray[FLOAT_t, ndim=2] Ox = coords[indexes_at_t]
    cdef np.ndarray[FLOAT_t, ndim=2] H1 = coords[indexes_at_t + 1]
    cdef np.ndarray[FLOAT_t, ndim=2] H2 = coords[indexes_at_t + 2]
    cdef np.ndarray[FLOAT_t, ndim=2] vectors = (H1 + H2) * 0.5 - Ox
    return vectors

def build_unit_vectors(np.ndarray[FLOAT_t, ndim=2] vectors):
    """
    Normalize a M x 3 matrix, M vectors (x, y, z) and return unit vectors

    Args:
        vectors (numpy.ndarray) : M x 3 array, M vectors (x, y, z)

    Returns:
        M x 3 matrix : A collection of unit vectors
    """
    cdef np.ndarray[FLOAT_t, ndim=1] nvectors = np.linalg.norm(vectors, axis=1)
    cdef np.ndarray[FLOAT_t, ndim=2] uvectors = vectors / nvectors[:,np.newaxis]
    return uvectors

def build_overlap_indexes(np.ndarray[INT_t, ndim=2] indexes,
                          int dt,
                          set exclude=set([-1])):
    """
    Build overlap indexes
    """
    cdef int t = 0
    cdef set lhs, rhs
    cdef list both
    cdef list overlaps = []
    #for t in range(0, len(indexes)-dt):
    for t in range(0, len(indexes)-dt, dt):
        lhs = set(indexes[t])
        rhs = set(indexes[t + dt])
        both = sorted(list(lhs & rhs - exclude))
        overlaps.append(np.array(both, dtype=np.int))
    return overlaps


cdef class OrientationalRelaxation(object):

    cdef object trajectory
    cdef list indexes
    cdef int nframes
    cdef int taumax

    def __init__(object self,
                 object trajectory,
                 np.ndarray[INT_t, ndim=2] indexes,
                 int t0=0, int tf=-1, int taumax=20):
        tf = len(trajectory) if tf < 0 else tf
        self.trajectory = trajectory[t0:tf]
        self.nframes    = len(self.trajectory)
        self.taumax      = taumax

        self.indexes = []
        for index in indexes[t0:tf]:
            self.indexes.append(index[index >= 0])

    def _calc_mean_relaxation_delta(object self,
                                    int t, int tau):
        cdef np.ndarray[INT_t, ndim=1] index_at_t = self.indexes[t]
        cdef np.ndarray[FLOAT_t, ndim=2] vectors1 = build_dipole_vectors(
            self.trajectory, index_at_t, t
        )
        cdef np.ndarray[FLOAT_t, ndim=2] vectors2 = build_dipole_vectors(
            self.trajectory, index_at_t, t + tau
        )
        cdef np.ndarray[FLOAT_t, ndim=2] uvectors1 = build_unit_vectors(vectors1)
        cdef np.ndarray[FLOAT_t, ndim=2] uvectors2 = build_unit_vectors(vectors2)
        cdef np.ndarray[FLOAT_t, ndim=1] lhs, rhs
        cdef list correlations = [
            lg2(np.dot(lhs, rhs)) for lhs, rhs in zip(uvectors1, uvectors2)
        ]
        return np.mean(correlations)

    def _calc_mean_relaxation(object self, int tau):
        cdef int t
        cdef list relaxation_deltas = [
            self._calc_mean_relaxation_delta(t, tau)
            for t in range(0, self.nframes-tau, tau)
        ]
        return np.mean(relaxation_deltas)

    def calc(object self):
        cdef int dt
        yield 1.0
        for tau in range(1, self.taumax):
            yield self._calc_mean_relaxation(tau)

def calc_orientational_relaxation(object trajectory,
                                  np.ndarray[INT_t, ndim=2] indexes,
                                  int t0=0, int tf=-1, int taumax=20):
    instance = OrientationalRelaxation(
        trajectory, indexes, t0, tf, taumax
    )
    return instance.calc()


cdef class SurvivalProbability(object):

    cdef object trajectory
    cdef np.ndarray indexes
    cdef int nframes
    cdef int taumax
    cdef set exclude

    def __init__(object self,
                 object trajectory,
                 np.ndarray[INT_t, ndim=2] indexes,
                 int t0=0, int tf=-1, int taumax=10):
        tf = len(trajectory) if tf < 0 else tf
        self.trajectory = trajectory[t0:tf]
        self.indexes = indexes[t0:tf]
        self.nframes = len(self.trajectory)
        self.taumax = taumax
        self.exclude = set([-1])

    def _count_stayed_atoms(object self, int t, int tau):
        cdef int dt
        cdef set stayed = set(self.indexes[t])
        for dt in range(1, min(tau, len(self.indexes)-t)):
            stayed &= set(self.indexes[t+dt])
        return len(stayed - self.exclude)

    def _calc_mean_survival_delta(object self, int t, int tau):
        cdef int Ntau = self._count_stayed_atoms(t, tau)
        cdef int Nt   = len(set(self.indexes[t]) - self.exclude)
        return 0 if Nt == 0 else float(Ntau)/float(Nt)

    def _calc_mean_survival(object self, int tau):
        cdef int t
        cdef list survival_deltas = list(filter(
            lambda float x: x != 0, (
                self._calc_mean_survival_delta(t, tau)
                for t in range(1, self.nframes-tau)
            )))
        return np.mean(survival_deltas)

    def calc(object self):
        cdef int tau
        for tau in range(1, self.taumax+1):
            yield self._calc_mean_survival(tau)

def calc_survival_probability(object trajectory,
                              np.ndarray[INT_t, ndim=2] indexes,
                              int t0=0, int tf=-1, int taumax=20):
    instance = SurvivalProbability(
        trajectory, indexes, t0, tf, taumax
    )
    return instance.calc()
