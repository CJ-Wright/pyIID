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


class ElasticScatter(object):
    """
    Scatter contains all the methods associated with producing theoretical
    diffraction patterns and PDFs from atomic configurations.  It does not
    include potential energies, such as Rw and chi**2 which are under the
    Calculator object.
    """

    def __init__(self, exp_dict=None):
        # Currently supported processor architectures
        self.avail_pro = ['MPI-GPU', 'Multi-GPU', 'CPU']
        self.exp = None
        self.pdf_qbin = None
        self.update_experiment(exp_dict)

        # Just in case something blows up down the line
        self.fq = cpu_wrap_fq
        self.grad = cpu_wrap_fq_grad
        self.processor = 'CPU'

        # Get the fastest processor architecture available
        self.set_processor()

        # Flag for scatter, if True update atoms
        self.scatter_needs_update = True

    def check_scatter(self, atoms):
        if self.scatter_needs_update is True \
                or 'exp' not in atoms.info.keys() \
                or atoms.info['exp'] != self.exp:
            wrap_atoms(atoms, self.exp)
            self.scatter_needs_update = False

    def update_experiment(self, exp_dict):
        # Should be read in from the gr file, but if not here are some defaults
        if exp_dict is None or bool(exp_dict) is False:
            exp_dict = {'qmin': 0.0, 'qmax': 25., 'qbin': .1, 'rmin': 0.0,
                        'rmax': 40.0, 'rstep': .01}
        # TODO: Check keys to make certain everything is there

        self.exp = exp_dict
        # Technically we should use this for qbin:
        self.pdf_qbin = np.pi / (self.exp['rmax'] + 6 * 2 * np.pi /
                                    self.exp['qmax'])
        self.scatter_needs_update = True

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
            from pyiid.wrappers.mpi_gpu_wrap import \
                wrap_fq as multi_node_gpu_wrap_fq
            from pyiid.wrappers.mpi_gpu_wrap import \
                wrap_fq_grad as multi_node_gpu_wrap_fq_grad

            self.fq = multi_node_gpu_wrap_fq
            self.grad = multi_node_gpu_wrap_fq_grad
            self.processor = processor
            return True

        elif processor == self.avail_pro[1] and check_multi_gpu() is True:
            from pyiid.wrappers.multi_gpu_wrap import \
                wrap_fq as node_0_gpu_wrap_fq
            from pyiid.wrappers.multi_gpu_wrap import \
                wrap_fq_grad as node_0_gpu_wrap_fq_grad

            from pyiid.wrappers.flat_gpu_wrap import wrap_fq as flat_fq
            from pyiid.wrappers.flat_gpu_wrap import \
                wrap_fq_grad as flat_grad

            # self.fq = node_0_gpu_wrap_fq
            # self.grad = node_0_gpu_wrap_fq_grad
            self.fq = flat_fq
            self.grad = flat_grad

            self.processor = processor
            return True

        elif processor == self.avail_pro[2]:
            self.fq = cpu_wrap_fq
            self.grad = cpu_wrap_fq_grad
            self.processor = processor
            return True

    def get_fq(self, atoms):
        self.check_scatter(atoms)
        return self.fq(atoms, self.exp['qbin'])

    def get_pdf(self, atoms):
        self.check_scatter(atoms)
        fq = self.fq(atoms, self.pdf_qbin, 'PDF')
        r = self.get_r()
        pdf0 = get_pdf_at_qmin(
            fq,
            self.exp['rstep'],
            self.pdf_qbin,
            r,
            self.exp['qmin']
        )

        return pdf0

    def get_sq(self, atoms):
        fq = self.get_fq(atoms)
        scatter_vector = np.arange(0, self.exp['qmax'], self.exp['qbin'])
        old_settings = np.seterr(all='ignore')
        sq = (fq / scatter_vector) + np.ones(scatter_vector.shape)
        np.seterr(**old_settings)
        sq[np.isinf(sq)] = 0.
        return sq

    def get_iq(self, atoms):
        return self.get_sq(atoms) * np.average(atoms.get_array('F(Q) scatter')) ** 2

    def get_grad_fq(self, atoms):
        self.check_scatter(atoms)
        return self.grad(atoms, self.exp['qbin'])

    def get_grad_pdf(self, atoms):
        self.check_scatter(atoms)
        fq_grad = self.grad(atoms, self.pdf_qbin, 'PDF')
        qmin_bin = int(self.exp['qmin'] / self.pdf_qbin)
        fq_grad[:, :, :qmin_bin] = 0.
        rgrid = self.get_r()

        pdf_grad = grad_pdf(fq_grad, self.exp['rstep'], self.pdf_qbin,
                            rgrid,
                            self.exp['qmin'])
        return pdf_grad

    def get_scatter_vector(self):
        return np.arange(self.exp['qmin'], self.exp['qmax'], self.exp['qbin'])

    def get_r(self):
        return np.arange(self.exp['rmin'], self.exp['rmax'], self.exp['rstep'])


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
    if 'qbin' not in exp_dict.keys():
        exp_dict['qbin'] = .1
    n = len(atoms)
    e_num = atoms.get_atomic_numbers()
    e_set = set(e_num)
    e_list = list(e_set)
    # F(Q) version
    qmax_bin = int(math.ceil(exp_dict['qmax'] / exp_dict['qbin']))
    set_scatter_array = np.zeros((len(e_set), qmax_bin), dtype=np.float32)
    get_scatter_array(set_scatter_array, e_num, exp_dict['qbin'])
    scatter_array = np.zeros((n, qmax_bin), dtype=np.float32)
    for i in range(len(e_set)):
        scatter_array[
        np.where(atoms.numbers == e_list[i])[0], :] = set_scatter_array[i, :]
    if 'F(Q) scatter' in atoms.arrays.keys():
        del atoms.arrays['F(Q) scatter']
    atoms.set_array('F(Q) scatter', scatter_array)

    # PDF version
    qbin = np.pi / (exp_dict['rmax'] + 6 * 2 * np.pi / exp_dict['qmax'])
    qmax_bin = int(math.ceil(exp_dict['qmax'] / qbin))
    set_scatter_array = np.zeros((len(e_set), qmax_bin), dtype=np.float32)
    get_scatter_array(set_scatter_array, e_num, qbin)
    scatter_array = np.zeros((n, qmax_bin), dtype=np.float32)
    for i in range(len(e_set)):
        scatter_array[
        np.where(atoms.numbers == e_list[i])[0], :] = set_scatter_array[i, :]
    if 'PDF scatter' in atoms.arrays.keys():
        del atoms.arrays['PDF scatter']
    atoms.set_array('PDF scatter', scatter_array)

    atoms.info['exp'] = exp_dict



if __name__ == '__main__':
    from ase.atoms import Atoms
    import ase.io as aseio
    import matplotlib.pyplot as plt

    atoms = aseio.read('/mnt/bulk-data/Dropbox/BNL_Project/Simulations/Models.d/2-AuNP-DFT.d/SizeVariation.d/Au55.initial_VASP_Oh.xyz',)
    # atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    # n = 1500
    # pos = np.random.random((n, 3)) * 10.
    # atoms = Atoms('Au' + str(n), pos)
    exp_dict = {
        'qmin': 0.0,
        'qmax': 25.,
        'qbin': .1,
        'rmin': 2.45,
        'rmax': 20.,
        'rstep': .01
    }
    scat = ElasticScatter(exp_dict)
    # fq = scat.get_fq(atoms)
    # gfq = scat.get_grad_fq(atoms)
    pdf = scat.get_pdf(atoms)
    r = scat.get_r()
    # gpdf = scat.get_grad_pdf(atoms)
    plt.plot(r, pdf)
    plt.show()
