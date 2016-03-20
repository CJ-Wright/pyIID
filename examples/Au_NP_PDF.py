from copy import deepcopy as dc
from ase import Atoms
from pyiid.experiments.elasticscatter import ElasticScatter
from pyiid.calc.calc_1d import Calc1D
from pyiid.sim.nuts_hmc import NUTSCanonicalEnsemble
from ase.cluster import Octahedron
import matplotlib.pyplot as plt
from ase.visualize import view


# Lets set up the atoms
# We use the Atomic Simulation Environment to take care of our atoms
# atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
atoms = Octahedron('Au', 2)
view(atoms)
# we can view the atoms by importing ASE's gui
scat = ElasticScatter()
pdf = scat.get_pdf(atoms)
r = scat.get_r()
# Now lets dilate the atoms so that they don't match the pdf
atoms2 = dc(atoms)
atoms2.positions *= 1.05
pdf2 = scat.get_pdf(atoms2)
r = scat.get_r()

# Now we need to define the potential energy surface
calc = Calc1D(
    target_data=pdf,  # The target or experimental data
    exp_function=scat.get_pdf,
    # The function which takes in atoms and produces
    # data like the experiment
    exp_grad_function=scat.get_grad_pdf,  # the function which produces the
    #  gradient of the calculated data
    conv=100,  # conversion from the unitless goodness of fit to eV
    potential='rw' # use the rw PES over chi squared
)

# Now we attach the calculator to our displaced atoms
atoms2.set_calculator(calc)
# Now we can get back the potential energy
print atoms2.get_potential_energy()
# And the forces
print atoms2.get_forces()

# Now we need to make the ensemble
ensemble = NUTSCanonicalEnsemble(atoms2, temperature=1000,
                                 verbose=True, escape_level=8)
# Now re run the simulation
traj, metadata = ensemble.run(20)
view(traj)