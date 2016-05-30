import mdtraj as md

def load_partial(netcdf, prmtop, start, stop, stride=1):
    topology = md.load_topology(prmtop)
    with md.open(netcdf) as f:
        f.seek(start)
        t = f.read_as_traj(
            topology,
            n_frames=int((stop-start)/stride),
            stride=stride,
        )
        return t
