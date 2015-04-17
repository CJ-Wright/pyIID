__author__ = 'christopher'
from ase.atoms import Atoms
from numba import cuda
import numpy as np
import mpi4py


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

        super(PDFAtoms, self).__init__(symbols, positions,
                                       numbers, tags, momenta,
                                       masses, magmoms, charges,
                                       scaled_positions, cell,
                                       pbc, celldisp,
                                       constraint, calculator,
                                       info)
        # Test which one of the kernels to use: MPI, Single Node-MulitGPU, Serial CPU
        for i in ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']:
            try:
                if i == 'MPI-GPU':
                        # TEST FOR MPI
                        pass
                elif i == 'Multi-GPU':
                    cuda.get_current_device()
                    self.processor = 'gpu'
                    from pyiid.wrappers.multi_gpu_wrap import wrap_fq, wrap_fq_grad
                    cuda.close()
                elif i == 'Serial-CPU':
                    self.processor = 'cpu'
                from pyiid.wrappers. import wrap_fq, wrap_fq_grad

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
        scatter_array = np.zeros((len(self.positions), self.qmax/self.qbin), dtype=np.float32)
        get_scatter_array(scatter_array, self.get_atomic_numbers(), self.qbin)
        self.new_array('scatter_array', scatter_array, np.float32)
        self.pdf = None
        self.fq = None

    def get_scatter_array(self):
        return self.arrays['scatter_array'].copy()

    def get_pdf(self):
        self.pdf, self.fq = wrap_pdf(self, qmax=self.qmax,
                 qmin=self.qmin, qbin=self.qbin, rmax=self.rmax,
                 rstep=self.rbin, fq_calc=self.fq_calc)
        return self.pdf

if __name__ == '__main__':
    from ase.visualize import view
    import matplotlib.pyplot as plt
    from pyiid.ase_ext.calc.pdfcalc import PDFCalc

    atoms = Atoms('Au4', [[0,0,0],[3,0,0],[0,3,0],[3,3,0]])
    pa = PDFAtoms(atoms)
    pa.get_pdf()
    ipdf = pa.pdf
    # plt.plot(ipdf), plt.show()
    pa.positions = [[-1,0,0],[3,0,0],[0,3,0],[3,3,0]]
    pa.set_calculator(PDFCalc(gobs=ipdf))
    print pa.get_potential_energy()
