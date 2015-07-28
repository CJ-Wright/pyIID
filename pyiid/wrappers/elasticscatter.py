__author__ = 'christopher'
import math

from numba import cuda
import numpy as np
from pyiid.kernels.master_kernel import grad_pdf as cpu_grad_pdf

from pyiid.kernels.master_kernel import get_pdf_at_qmin, get_scatter_array
from pyiid.wrappers.cpu_wrappers.cpu_wrap import wrap_fq as cpu_wrap_fq
from pyiid.wrappers.cpu_wrappers.cpu_wrap import \
    wrap_fq_grad as cpu_wrap_fq_grad


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
        self.exp_dict_keys = ['qmin', 'qmax', 'qbin', 'rmin', 'rmax', 'rstep',
                              'sampling']
        self.default_values = [0.0, 25, .1, 0.0, 40.0, .01, 'full']
        self.alg = None
        self.exp = None
        self.pdf_qbin = None
        self.update_experiment(exp_dict)

        # Just in case something blows up down the line
        self.fq = cpu_wrap_fq
        self.grad = cpu_wrap_fq_grad
        self.grad_pdf = cpu_grad_pdf
        self.processor = 'CPU'
        self.alg = 'nxn'

        # Get the fastest processor architecture available
        self.set_processor()

        # Flag for scatter, if True update atoms
        self.scatter_needs_update = True

    def set_processor(self, processor=None, kernel_type='flat'):
        """
        Set the processor to use for calculating the scattering.  If no
        parameter is given then check for the fastest possible processor
        configuration

        Parameters
        -----------
        processor: ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']
            The processor to use
        """
        # If a processor is given try to use that processor,
        # but check if it is viable first.

        if processor is None:
            # Test each processor in order of most advanced to least
            for pro in self.avail_pro:
                if self.set_processor(
                        processor=pro, kernel_type=kernel_type) is not None:
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
            if kernel_type == 'nxn':
                from pyiid.wrappers.multi_gpu_wrap import \
                    wrap_fq as node_0_gpu_wrap_fq
                from pyiid.wrappers.multi_gpu_wrap import \
                    wrap_fq_grad as node_0_gpu_wrap_fq_grad

                self.fq = node_0_gpu_wrap_fq
                self.grad = node_0_gpu_wrap_fq_grad
                self.alg = 'nxn'

            elif kernel_type == 'flat':
                from pyiid.wrappers.flat_gpu_wrap import wrap_fq as flat_fq
                from pyiid.wrappers.flat_gpu_wrap import \
                    wrap_fq_grad as flat_grad

                self.fq = flat_fq
                self.grad = flat_grad
                self.alg = 'flat'
            from pyiid.kernels.experimental_kernels import grad_pdf

            self.grad_pdf = grad_pdf
            self.processor = processor
            return True

        elif processor == self.avail_pro[2]:
            if kernel_type == 'nxn':
                self.fq = cpu_wrap_fq
                self.grad = cpu_wrap_fq_grad
                self.alg = 'nxn'

            elif kernel_type == 'flat':
                from pyiid.wrappers.cpu_wrappers.flat_multi_cpu_wrap import \
                    wrap_fq, wrap_fq_grad

                self.fq = wrap_fq
                self.grad = wrap_fq_grad
                self.alg = 'flat'

            self.grad_pdf = cpu_grad_pdf
            self.processor = processor
            return True

    def check_scatter(self, atoms):
        if self.scatter_needs_update is True \
                or 'exp' not in atoms.info.keys() \
                or atoms.info['exp'] != self.exp:
            wrap_atoms(atoms, self.exp)
            self.scatter_needs_update = False

    def update_experiment(self, exp_dict):
        # Should be read in from the gr file, but if not here are some defaults
        if exp_dict is None or bool(exp_dict) is False:
            exp_dict = {}
        for key, dv in zip(self.exp_dict_keys, self.default_values):
            if key not in exp_dict.keys():
                exp_dict[key] = dv
        if exp_dict['sampling'] == 'ns':
            exp_dict['rstep'] = np.pi / exp_dict['qmax']
        self.exp = exp_dict
        # Technically we should use this for qbin:
        self.pdf_qbin = np.pi / (self.exp['rmax'] + 6 * 2 * np.pi /
                                 self.exp['qmax'])
        self.scatter_needs_update = True

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
        return self.get_sq(atoms) * np.average(
            atoms.get_array('F(Q) scatter')) ** 2

    def get_2d_scatter(self, atoms, pixel_array):
        iq = self.get_iq(atoms)
        s = self.get_scatter_vector()
        qb = self.exp['qbin']
        final_shape = pixel_array.shape
        fp = pixel_array.ravel()
        img = np.zeros(fp.shape)
        for sub_s, i in zip(s, iq):
            c = np.intersect1d(np.where(sub_s - qb / 2. < fp)[0],
                               np.where(sub_s + qb / 2. > fp)[0])
            img[c] = i
        return img.reshape(final_shape)

    def get_grad_fq(self, atoms):
        self.check_scatter(atoms)
        return self.grad(atoms, self.exp['qbin'])

    def get_grad_pdf(self, atoms):
        self.check_scatter(atoms)
        fq_grad = self.grad(atoms, self.pdf_qbin, 'PDF')
        qmin_bin = int(self.exp['qmin'] / self.pdf_qbin)
        fq_grad[:, :, :qmin_bin] = 0.
        rgrid = self.get_r()

        pdf_grad = self.grad_pdf(fq_grad, self.exp['rstep'], self.pdf_qbin,
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
