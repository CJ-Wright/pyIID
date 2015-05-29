__author__ = 'christopher'
from numba import cuda
import numpy as np
import math
import sys
sys.path.extend(['/mnt/work-data/dev/pyIID'])

from pyiid.kernels.master_kernel import get_pdf_at_qmin, grad_pdf, \
    get_scatter_array

from pyiid.wrappers.cpu_wrap import wrap_fq as cpu_wrap_fq
from pyiid.wrappers.cpu_wrap import wrap_fq_grad as cpu_wrap_fq_grad


def wrap_atoms(atoms, exp_dict=None):
    """
    Call this function before applying calculator, it will generate static
    arrays for the scattering, preventing recalculation
    :param exp_dict:
    :param atoms:
    :param qbin:
    :return:
    """

    if exp_dict is None:
        exp_dict = {'qmin': 0.0, 'qmax': 25., 'qbin': .1, 'rmin': 0.0,
                            'rmax': 40.0, 'rstep': .01}
    qbin = np.pi / (exp_dict['rmax'] + 6 * 2 * np.pi / exp_dict['qmax'])
    n = len(atoms)
    qmax_bin = int(math.ceil(exp_dict['qmax'] / qbin))
    e_num = atoms.get_atomic_numbers()

    scatter_array = np.zeros((n, qmax_bin), dtype=np.float32)
    get_scatter_array(scatter_array, e_num, qbin)
    atoms.set_array('scatter', scatter_array)

def check_mpi():
    # Test if MPI GPU is viable
    # Currently no working MPI GPU implementation
    return False


def check_multi_gpu():
    try:
        cuda.gpus.lst
        return True
    except:
        return False


class Scatter(object):
    """
    Scatter contains all the methods associated with producing theoretical
    diffraction patterns and PDFs from atomic configurations.  It does not
    include potential energies, such as Rw and chi**2 which are under the
    Calculator object.
    """

    def __init__(self, exp_dict=None):
        # Currently supported processor architectures
        self.avail_pro = ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']

        # Should be read in from the gr file, but if not here are some defaults
        if exp_dict is None or bool(exp_dict) is False:
            exp_dict = {'qmin': 0.0, 'qmax': 25., 'qbin': .1, 'rmin': 0.0,
                        'rmax': 40.0, 'rstep': .01}
        # Check keys


        self.exp = exp_dict
        # Technically we should use this for qbin:
        self.exp['qbin'] = np.pi / (self.exp['rmax'] + 6 * 2 * np.pi /
                                    self.exp['qmax'])

        # Just in case something blows up down the line
        self.fq = cpu_wrap_fq
        self.grad = cpu_wrap_fq_grad
        self.processor = 'Serial-CPU'

        # Get the fastest processor architecture available
        self.set_processor()

    def set_processor(self, processor=None):
        """
        Set the processor to use for calculating the scattering.  If no
        parameter is given then check for the fastest possible processor
        configuration

        Parameters
        -----------
        processor: ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']
            The processor to use
        """
        # If a processor is given try to use that processor, but check if it is
        # viable first.

        if processor is None:
            # Test each processor in order of most advanced to least
            for pro in self.avail_pro:
                if self.set_processor(processor=pro) is not None:
                    break

        elif processor == self.avail_pro[0] and check_mpi() is True:
            from pyiid.wrappers.mpi_gpu_wrap import wrap_fq as multi_node_gpu_wrap_fq
            from pyiid.wrappers.mpi_gpu_wrap import \
                wrap_fq_grad as multi_node_gpu_wrap_fq_grad

            self.fq = multi_node_gpu_wrap_fq
            self.grad = multi_node_gpu_wrap_fq_grad
            self.processor = processor

        elif processor == self.avail_pro[1] and check_multi_gpu() is True:
            from pyiid.wrappers.multi_gpu_wrap import wrap_fq as node_0_gpu_wrap_fq
            from pyiid.wrappers.multi_gpu_wrap import \
                wrap_fq_grad as node_0_gpu_wrap_fq_grad


            from pyiid.wrappers.flat_gpu_wrap import wrap_fq as flat_fq
            from pyiid.wrappers.flat_gpu_wrap import \
                wrap_fq_grad as flat_grad

            # self.fq = node_0_gpu_wrap_fq
            self.grad = node_0_gpu_wrap_fq_grad

            self.fq = flat_fq
            # self.grad = flat_grad
            self.processor = processor

        elif processor == self.avail_pro[2]:
            self.fq = cpu_wrap_fq
            self.grad = cpu_wrap_fq_grad
            self.processor = processor


    # def wrap_check(self, atoms):
    #     if atoms.get_array('scatter') is None:


    def get_fq(self, atoms):
        return self.fq(atoms, self.exp['qmax'], self.exp['qbin'])

    def get_pdf(self, atoms):
        fq = self.fq(atoms, self.exp['qmax'], self.exp['qbin'])
        # fq[:int(self.exp['qmin'] / self.exp['qbin'])] = 0
        r = np.arange(self.exp['rmin'], self.exp['rmax'], self.exp['rstep'])
        pdf0 = get_pdf_at_qmin(
            fq,
            self.exp['rstep'],
            self.exp['qbin'],
            r,
            self.exp['qmin']
        )

        # pdf = pdf0[int(self.exp['rmin'] / self.exp['rstep']):int(self.exp['rmax'] / self.exp['rstep'])]
        return pdf0

    def get_sq(self, atoms):
        fq = self.fq(atoms, self.exp['qmax'], self.exp['qbin'])
        scatter_vector = np.arange(0, self.exp['qmax'], self.exp['qbin'])
        sq = (fq / scatter_vector) + np.ones(scatter_vector.shape)
        sq[np.isinf(sq)] = 0.
        return sq

    def get_iq(self, atoms):
        return self.get_sq(atoms) * np.average(atoms.get_array('scatter')) ** 2

    def get_grad_fq(self, atoms):
        return self.grad(atoms, self.exp['qmax'], self.exp['qbin'])

    def get_grad_pdf(self, atoms):
        fq_grad = self.grad(atoms, self.exp['qmax'], self.exp['qbin'])
        qmin_bin = int(self.exp['qmin'] / self.exp['qbin'])
        fq_grad[:, :, :qmin_bin] = 0.
        rgrid = np.arange(self.exp['rmin'], self.exp['rmax'],
                          self.exp['rstep'])
        pdf_grad = np.zeros(
            (len(atoms), 3, len(rgrid)))

        grad_pdf(
            pdf_grad,
            fq_grad,
            self.exp['rstep'],
            self.exp['qbin'],
            rgrid,
            self.exp['qmin']
        )
        return pdf_grad


if __name__ == '__main__':
    from ase.atoms import Atoms
    import matplotlib.pyplot as plt

    # atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    n = 1500
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au' + str(n), pos)
    exp_dict = {'qmin': 0.0, 'qmax': 25.,
                'qbin': np.pi / (45. + 6 * 2 * np.pi / 25), 'rmin': 0.0,
                'rmax': 40.0, 'rstep': .01}
    wrap_atoms(atoms, exp_dict)
    scat = Scatter(exp_dict)
    scat.set_processor('Multi-GPU')
    gfq = scat.get_grad_fq(atoms)
    fq = scat.get_fq(atoms)