#******************************************************************************
#
# mdtraj-contrib
# Water dynamics analysis module
#
# Author:  lambdalisue
# License: GNU Public License v3 (Inherited from the original)
#
# NOTE:
# Written code here are based on MDAnalysis.analysis.waterdynamics written by
# Alejandro Bernardin.
# A lot of modifications and aggressive refactorings were performed by
# lambdalisue to make it faster and compatible with MDTraj
#
#******************************************************************************
import numpy as np
import mdtraj as md
cimport numpy as np


def lg2(float x):
    """
    Second Legendre polynomial
    Ref: https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    return (3 * x**2 - 1) / 2.

def build_dipole_vectors(trajectory, np.ndarray indexes, int t):
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
    cdef np.ndarray coords = trajectory.xyz[t]
    cdef np.ndarray Ox = coords[indexes]
    cdef np.ndarray H1 = coords[indexes + 1]
    cdef np.ndarray H2 = coords[indexes + 2]
    cdef np.ndarray vectors = (H1 + H2) * 0.5 - Ox
    return vectors

def build_unit_vectors(np.ndarray vectors):
    """
    Normalize a M x 3 matrix, M vectors (x, y, z) and return unit vectors

    Args:
        vectors (numpy.ndarray) : M x 3 array, M vectors (x, y, z)

    Returns:
        M x 3 matrix : A collection of unit vectors
    """
    cdef np.ndarray nvectors = np.linalg.norm(vectors, axis=1)
    cdef np.ndarray uvectors = vectors / nvectors[:,np.newaxis]
    return uvectors

def build_overlap_indexes(np.ndarray indexes, int dt,
                          exclude=set([-1])):
    """
    Build overlap indexes
    """
    cdef int t = 0
    cdef set lhs, rhs
    cdef list both
    cdef list overlaps = []
    for t in range(0, len(indexes)-dt):
        lhs = set(indexes[t])
        rhs = set(indexes[t + dt])
        both = sorted(list(lhs & rhs - exclude))
        overlaps.append(np.array(both))
    return overlaps


class OrientationalRelaxation(object):
    def __init__(self, trajectory, np.ndarray indexes,
                 int t0=0, int tf=-1, int dtmax=20):
        tf = len(trajectory) if tf < 0 else tf
        self.trajectory = trajectory[t0:tf]
        self.indexes    = indexes[t0:tf]
        self.nframes    = len(self.trajectory)

    def _build_overlap_indexes(self, int dt):
        if not hasattr(self, '_cached_overlap_indexes'):
            self._cached_overlap_indexes = {}
        if dt not in self._cached_overlap_indexes:
            indexes = build_overlap_indexes(self.indexes, dt)
            self._cached_overlap_indexes[dt] = indexes
        return self._cached_overlap_indexes[dt]

    def _calc_mean_relaxation_delta(self, list overlap_indexes, int dt, int t):
        cdef np.ndarray indexes = overlap_indexes[int(t/dt)]
        cdef np.ndarray vectors1 = build_dipole_vectors(
            self.trajectory, indexes, t
        )
        cdef np.ndarray vectors2 = build_dipole_vectors(
            self.trajectory, indexes, t + dt
        )
        cdef np.ndarray uvectors1 = build_unit_vectors(vectors1)
        cdef np.ndarray uvectors2 = build_unit_vectors(vectors2)
        cdef list correlations = [
            lg2(np.dot(lhs, rhs)) for lhs, rhs in zip(uvectors1, uvectors2)
        ]
        return np.mean(correlations)

    def _calc_mean_relaxation(self, int dt):
        cdef list overlap_indexes = self._build_overlap_indexes(dt)
        cdef list relaxation_deltas = [
            self._calc_mean_relaxation_delta(overlap_indexes, dt, t)
            for t in range(0, self.nframes, dt)
        ]
        return np.mean(relaxation_deltas)

    def calc(self):
        cdef int dt
        yield 1.0
        for dt in range(1, self.dtmax):
            yield self._calc_mean_relaxation(dt)

def calc_orientational_relaxation(trajectory, np.ndarray indexes,
                                  int t0=0, int tf=-1, int dtmax=20):
    instance = OrientationalRelaxation(
        trajectory, indexes, t0, tf, dtmax
    )
    return instance.calc()
