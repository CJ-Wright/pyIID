import numpy as np
from ase.atoms import Atoms
from pyiid.experiments.elasticscatter import ElasticScatter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ase.io
__author__ = 'christopher'

# atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
atoms = ase.io.read('/mnt/work-data/dev/IID_data/db_test/PDF_LAMMPS_587.traj')
scat = ElasticScatter()
scat.update_experiment({'qmax': 8, 'qmin': .5})
s = scat.get_scatter_vector()

k = 100
i_max, j_max = k, k

pixel_array = np.zeros((i_max, j_max))
for i in range(i_max):
    for j in range(j_max):
        pixel_array[i, j] = np.sqrt(i ** 2 + j ** 2)

pixel_array /= np.max(pixel_array)
pixel_array *= np.max(s)

img = scat.get_2d_scatter(atoms, pixel_array)
print(img.shape)


# plt.imshow(pixel_array)
print(np.max(img), np.min(img))
plt.imshow(img, aspect='auto')
plt.colorbar()
plt.show()

plt.plot(scat.get_iq(atoms))
plt.show()
