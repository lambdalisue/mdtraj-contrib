import numpy as np
from mdtraj.utils import import ensure_type

def compute_distances_from(traj, coordinate, atoms, periodic=True, opt=True):
    xyz = ensure_type(
        traj.xyz, dtype=np.float32, ndim=3,
        name='traj.xyz', shape=(None, None, 3), warn_on_cast=False,
    )
    coordinate = ensure_type(
        coordinate, dtype=np.float32, ndim=1,
        name='coordinate', shape=(3,), warn_on_cast=False,
    )
    atoms = ensure_type(
        atoms, dtype=np.int32, ndim=1,
        name='atoms', shape=(None,), warn_on_cast=False,
    )
    if not np.all(np.logical_and(atoms < traj.n_atoms, atoms >= 0)):
        raise ValueError('atoms must be between 0 and %d' % traj.n_atoms)

    if len(atoms) == 0:
        return np.zeros((len(xyz), 0), dtype=np.float32)

    if periodic and traj._have_unitcell:
        box = ensure_type(
            traj.unitcell_vectors, dtype=np.float32, ndim=3,
            name='unitcell_vectors', shape=(len(xyz), 3, 3),
            warn_on_cast=False
        )
        orthogonal = np.allclose(traj.unitcell_angles, 90)
        return _distance_from_mic(
            xyz, pairs, box.transpose(0, 2, 1), orthogonal
        )
    return _distance_from(xyz, coordinate, atoms)

def _distance_from(xyz, coordinate, atoms):
    "Distance between coordinate and atoms in each frame"
    delta = np.diff([coordinate, xyz[:,atoms]], axis=0)[0]
    return (delta ** 2.).sum(-1) ** 0.5

def _distance_mic(xyz, coordinate, atoms, box_vectors, orthogonal):
    # WIP
    #out = np.empty((xyz.shape[0], pairs.shape[0]), dtype=np.float32)
    #for i in range(len(xyz)):
    #    hinv = np.linalg.inv(box_vectors[i])
    #    bv1, bv2, bv3 = box_vectors[i].T

    #    for j, (a,b) in enumerate(pairs):
    #        s1 = np.dot(hinv, xyz[i,a,:])
    #        s2 = np.dot(hinv, xyz[i,b,:])
    #        s12 = s2 - s1

    #        s12 = s12 - np.round(s12)
    #        r12 = np.dot(box_vectors[i], s12)
    #        dist = np.linalg.norm(r12)
    #        if not orthogonal:
    #            for ii in range(-1, 2):
    #                v1 = bv1*ii
    #                for jj in range(-1, 2):
    #                    v12 = bv2*jj + v1
    #                    for kk in range(-1, 2):
    #                        new_r12 = r12 + v12 + bv3*kk
    #                        dist = min(dist, np.linalg.norm(new_r12))
    #        out[i, j] = dist
    #return out
