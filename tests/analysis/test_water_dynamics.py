from nose.tools import *
import pyximport; pyximport.install()
from mdtraj_contrib.analysis.water_dynamics import *
import numpy as np

def test_build_overlap_indexes():
    indexes = np.array([
        [1, 2, 3, 4, 5, 0],
        [1, 2, 3, 4, 5, 0],
        [1, 0, 0, 4, 0, 0],
        [0, 0, 0, 4, 0, 6],
        [0, 2, 0, 0, 0, 6],
        [0, 2, 0, 0, 0, 6],
        [1, 2, 3, 4, 0, 0],
        [1, 0, 3, 4, 0, 0],
        [1, 0, 3, 4, 0, 0],
        [1, 0, 3, 0, 0, 0],
    ])
    indexes = indexes
    overlap_indexes = build_overlap_indexes(indexes, 1, exclude=set([0]))
    overlap_indexes = list(map(lambda x: tuple(x), overlap_indexes))
    eq_(overlap_indexes, [
        (1, 2, 3, 4, 5),
        (1, 4),
        (4,),
        (6,),
        (2, 6),
        (2,),
        (1, 3, 4),
        (1, 3, 4),
        (1, 3),
    ])

    overlap_indexes = build_overlap_indexes(indexes, 2, exclude=set([0]))
    overlap_indexes = list(map(lambda x: tuple(x), overlap_indexes))
    eq_(overlap_indexes, [
        (1, 4),
        (),
        (2,),
        (1, 3, 4),
    ])

    overlap_indexes = build_overlap_indexes(indexes, 3, exclude=set([0]))
    overlap_indexes = list(map(lambda x: tuple(x), overlap_indexes))
    eq_(overlap_indexes, [
        (4,),
        (4,),
        (1, 3),
    ])
