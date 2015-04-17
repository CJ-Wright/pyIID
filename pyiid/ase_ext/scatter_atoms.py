__author__ = 'christopher'
from ase.atoms import Atoms
from numba import cuda
import numpy as np
import os


class PDFAtoms(Atoms):
    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None, qmin=0.0, qmax=25.0, qbin=.1,
                 rmin=0.0, rmax=40.0, rbin=.01):

        # Init the Atoms method to get all the variables/methods from Atoms
        super(PDFAtoms, self).__init__(symbols, positions,
                                       numbers, tags, momenta,
                                       masses, magmoms, charges,
                                       scaled_positions, cell,
                                       pbc, celldisp,
                                       constraint, calculator,
                                       info)

        # Test which kernel to use: MPI, Single Node-MultiGPU, Serial CPU
        for i in ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']:
            try:
                if i == 'MPI-GPU':
                    # TEST FOR MPI
                    assert os.getenv('OMPI_COMM_WORLD_SIZE') is not None
                    from pyiid.wrappers.mpi_gpu_wrap import wrap_fq, \
                        wrap_fq_grad_gpu
                elif i == 'Multi-GPU':
                    cuda.get_current_device()
                    self.processor = 'gpu'
                    from pyiid.wrappers.multi_gpu_wrap import wrap_fq, \
                        wrap_fq_grad
                    cuda.close()
                elif i == 'Serial-CPU':
                    self.processor = 'cpu'
                    from pyiid.wrappers.serial_cpu_wrap import wrap_fq, \
                        wrap_fq_grad
            except:
                continue

        self.fq_grad_calc = wrap_fq_grad
        self.fq_calc = wrap_fq
        self.qmin = qmin
        self.qmax = qmax
        if qbin is None:
            self.qbin = np.pi / (rmax + 6 * 2 * np.pi / qmax)
        else:
            self.qbin = qbin
        self.rmin = rmin
        self.rmax = rmax
        self.rmin = rmin
        self.rbin = rbin
        scatter_array = np.zeros((len(self.positions), self.qmax / self.qbin),
                                 dtype=np.float32)
        get_scatter_array(scatter_array, self.get_atomic_numbers(), self.qbin)
        self.new_array('scatter_array', scatter_array, np.float32)
        self.pdf = None
        self.fq = None

    def get_scatter_array(self):
        return self.arrays['scatter_array'].copy()

    def get_pdf(self):
        self.pdf, self.fq = wrap_pdf(self, qmax=self.qmax,
                                     qmin=self.qmin, qbin=self.qbin,
                                     rmax=self.rmax,
                                     rstep=self.rbin, fq_calc=self.fq_calc)
        return self.pdf


if __name__ == '__main__':