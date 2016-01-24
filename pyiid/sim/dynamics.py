from pyiid.sim import leapfrog
__author__ = 'christopher'


def classical_dynamics(atoms, stepsize, n_steps):
    """
    Create a new atomic configuration by simulating the hamiltonian dynamics
    of the system

    Parameters
    ----------
    atoms: ase.Atoms ase.Atoms
        The atomic configuration
    stepsize: float
        The step size for the simulation
    n_steps: int
        The number of steps

    Returns
    -------
         list of ase.Atoms
        This list contains all the moves made during the simulation
    """
    atoms.get_forces()
    traj = [atoms]
    for n in range(n_steps):
        traj.append(leapfrog(traj[-1], stepsize))
    return traj
