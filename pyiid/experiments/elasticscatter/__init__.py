"""
The main class in this module `ElasticScatter` holds the experimental details,
and processor information needed to calculate the elastic powder scattering
from a collection of atoms.
"""
import math
from ase.units import s
import numpy as np
from numba import cuda

from .cpu_wrappers.nxn_cpu_wrap import \
    (wrap_fq_grad as cpu_wrap_fq_grad,
     wrap_fq as cpu_wrap_fq,
     wrap_voxel_fq as cpu_wrap_voxel_fq)
from .kernels.master_kernel import \
    (grad_pdf as cpu_grad_pdf,
     voxel_pdf as cpu_wrap_voxel_pdf,
     get_pdf_at_qmin, get_scatter_array)
from .kernels.cpu_flat import get_normalization_array as flat_norm
from ase.calculators.calculator import equal

__author__ = 'christopher'

all_changes = ['positions', 'numbers', 'cell', 'pbc', 'charges', 'magmoms',
               'exp']


def check_mpi():
    # Test if MPI GPU is viable
    # Currently no working MPI GPU implementation
    return False


def check_gpu():
    """
    Check if GPUs are available on this machine
    """
    try:
        cuda.gpus.lst
        tf = True
    except cuda.CudaSupportError:
        tf = False
    return tf


def check_cudafft():
    try:
        from numbapro.cudalib import cufft
        tf = True
    except ImportError:
        tf = False
        print 'no cudafft'
    return tf


def wrap_atoms(atoms, exp_dict=None):
    """
    Call this function before applying calculator, it will generate static
    arrays for the scattering, preventing recalculation

    Parameters
    -----------
    atoms: ase.Atoms
        The atoms to which scatter factors are added
    exp_dict: dict or None
        The experimental parameters, if None defaults are used
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
    atoms.info['scatter_atoms'] = n


class ElasticScatter(object):
    """
    Scatter contains all the methods associated with producing theoretical
    diffraction patterns and PDFs from atomic configurations.  It does not
    include potential energies, such as Rw and chi**2, which are under the
    Calculator object.
    >>>from ase.atoms import Atoms
    >>>import matplotlib.pyplot as plt
    >>>atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    >>>a = np.random.random(atoms.positions.shape) * .1
    >>>s = ElasticScatter({'rmax': 5., 'rmin': 2.})
    >>>fq = s.get_pdf(atoms)
    >>>fq2 = s.get_pdf(atoms)
    >>>plt.plot(s.get_r(), fq)
    >>>plt.show()

    """

# Internal Utilities ----------------------------------------------------------
    def __init__(self, exp_dict=None, verbose=False):
        self.atoms = None  # keep a copy of the atoms to prevent replication
        self.fq_result = None
        self.pdf_result = None
        self.fq_grad_result = None
        self.pdf_grad_result = None
        self.verbose = verbose

        # Currently supported processor architectures, in order of most
        # advanced to least
        self.avail_pro = ['MPI-GPU', 'Multi-GPU', 'CPU']

        # needed parameters to specify an experiment
        self.exp_dict_keys = ['qmin', 'qmax', 'qbin', 'rmin', 'rmax', 'rstep',
                              'sampling']
        # default experimental parameters
        self.default_values = [0.0, 25, .1, 0.0, 40.0, .01, 'full']
        # Initiate the algorithm, processor, and experiments
        self.alg = None
        self.processor = None
        self.exp = None
        self.pdf_qbin = None

        # set the experimental parameters
        self.update_experiment(exp_dict)

        # Just in case something blows up down the line set to the most base
        # processor
        self.fq = cpu_wrap_fq
        self.grad = cpu_wrap_fq_grad
        self.grad_pdf = cpu_grad_pdf
        self.voxel_fq = cpu_wrap_voxel_fq
        self.voxel_pdf = cpu_wrap_voxel_pdf
        self.processor = 'CPU'
        self.alg = 'nxn'

        # Get the fastest processor architecture available
        self.set_processor()

    def set_processor(self, processor=None, kernel_type='flat'):
        """
        Set the processor to use for calculating the scattering.  If no
        parameter is given then check for the fastest possible processor
        configuration

        Parameters
        -----------
        processor: ['MPI-GPU', 'Multi-GPU', 'Serial-CPU']
            The processor to use
        kernel_type: ['nxn', 'flat-serial', 'flat']
            The type of algorithm to use

        Returns
        -------
        bool:
            True on successful setup of the algorithm and processor
        """
        # If a processor is given try to use that processor,
        # but check if it is viable first.

        # Changing the processor invalidates the previous results
        self.fq_result = None
        self.pdf_result = None

        if processor is None:
            # Test each processor in order of most advanced to least
            for pro in self.avail_pro:
                if self.set_processor(
                        processor=pro, kernel_type=kernel_type) is not None:
                    break

        # MPI
        elif processor == self.avail_pro[0] and check_mpi() is True:
            from pyiid.experiments.elasticscatter.mpi_wrappers.mpi_gpu_wrap \
                import \
                wrap_fq as multi_node_gpu_wrap_fq
            from pyiid.experiments.elasticscatter.mpi_wrappers.mpi_gpu_wrap \
                import \
                wrap_fq_grad as multi_node_gpu_wrap_fq_grad

            self.fq = multi_node_gpu_wrap_fq
            self.grad = multi_node_gpu_wrap_fq_grad
            self.processor = processor
            return True

        #GPU
        elif processor == self.avail_pro[1] and check_gpu() is True:
            from pyiid.experiments.elasticscatter.gpu_wrappers.gpu_wrap import \
                (wrap_fq as flat_fq, wrap_fq_grad as flat_grad , wrap_voxel_fq)
            self.fq = flat_fq
            self.grad = flat_grad
            # self.voxel_fq = wrap_voxel_fq
            self.alg = 'flat'
            if check_cudafft():
                from pyiid.experiments.elasticscatter.gpu_wrappers.gpu_wrap import \
                    grad_pdf
                self.grad_pdf = grad_pdf
            else:
                self.grad_pdf = cpu_grad_pdf
            self.processor = processor
            return True
        # CPU
        elif processor == self.avail_pro[2]:
            if kernel_type == 'nxn':
                self.fq = cpu_wrap_fq
                self.grad = cpu_wrap_fq_grad
                self.alg = 'nxn'

            elif kernel_type == 'flat':
                from pyiid.experiments.elasticscatter.cpu_wrappers \
                    .flat_multi_cpu_wrap import \
                    wrap_fq, wrap_fq_grad, wrap_voxel_fq

                self.fq = wrap_fq
                self.grad = wrap_fq_grad
                self.voxel_fq = wrap_voxel_fq
                self.alg = 'flat'

            elif kernel_type == 'flat-serial':
                from pyiid.experiments.elasticscatter.cpu_wrappers \
                    .flat_serial_cpu_wrap import \
                    wrap_fq, wrap_fq_grad, wrap_voxel_fq

                self.fq = wrap_fq
                self.grad = wrap_fq_grad
                self.voxel_fq = wrap_voxel_fq
                self.alg = 'flat-serial'

            self.grad_pdf = cpu_grad_pdf
            self.processor = processor
            return True

    def update_experiment(self, exp_dict):
        """
        Change the scattering experiment parameters.

        Parameters
        ----------
        exp_dict: dict or None
            Dictionary of parameters to be updated, if None use defaults
        """
        # Should be read in from the gr file, but if not here are some defaults
        if exp_dict is None or bool(exp_dict) is False:
            exp_dict = {}
        for key, dv in zip(self.exp_dict_keys, self.default_values):
            if key not in exp_dict.keys():
                exp_dict[key] = dv

        # If sampling is ns then generate the PDF at
        # the Nyquist Shannon Sampling Frequency
        if exp_dict['sampling'] == 'ns':
            exp_dict['rstep'] = np.pi / exp_dict['qmax']

        self.exp = exp_dict
        # Technically we should use this for qbin
        self.pdf_qbin = np.pi / (self.exp['rmax'] + 6 * 2 * np.pi /
                                 self.exp['qmax'])

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        if self.atoms is None:
            system_changes = all_changes
        else:
            system_changes = []
            if not equal(self.atoms.positions, atoms.positions, tol):
                system_changes.append('positions')
            if not equal(self.atoms.numbers, atoms.numbers):
                system_changes.append('numbers')
            if not equal(self.atoms.cell, atoms.cell, tol):
                system_changes.append('cell')
            if not equal(self.atoms.pbc, atoms.pbc):
                system_changes.append('pbc')
            if not equal(self.atoms.get_initial_magnetic_moments(),
                         atoms.get_initial_magnetic_moments(), tol):
                system_changes.append('magmoms')
            if not equal(self.atoms.get_initial_charges(),
                         atoms.get_initial_charges(), tol):
                system_changes.append('charges')
            if 'exp' not in atoms.info.keys() \
                    or atoms.info['exp'] != self.exp \
                    or atoms.info['scatter_atoms'] != len(atoms) \
                    or True in np.all(atoms.arrays['F(Q) scatter'] == 0., 1) \
                    or True in np.all(atoms.arrays['PDF scatter'] == 0., 1):
                system_changes.append('exp')
        if self.verbose:
            print 'check_state results:', system_changes
        return system_changes

    def get_scatter_vector(self):
        """
        Calculate the scatter vector Q for the current experiment

        Returns
        -------
        1darray:
            The Q range for this experiment
        """
        return np.arange(self.exp['qmin'], self.exp['qmax'], self.exp['qbin'])

    def get_r(self):
        """
        Calculate the inter-atomic distance range for the current experiment

        Returns
        -------
        1darray:
            The r range for this experiment
        """
        return np.arange(self.exp['rmin'], self.exp['rmax'], self.exp['rstep'])

# Get experimental simulation results -----------------------------------------
    def get_fq(self, atoms):
        """
        Calculate the reduced structure factor F(Q)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate F(Q)
        Returns
        -------
        1darray:
            The reduced structure factor
        """
        state = self.check_state(atoms)
        if state == [] and self.fq_result is not None:
            if self.verbose:
                print 'using previous fq'
            return self.fq_result
        else:
            if any(x in ['exp', 'numbers'] for x in state):
                if self.verbose:
                    print 'calculating new scatter factors'
                wrap_atoms(atoms, self.exp)
                # Changing/Updating the scatter factors invalidates saved data
                self.fq_result = None
                self.pdf_result = None
                self.fq_grad_result = None
                self.pdf_grad_result = None
            fq = self.fq(atoms, self.exp['qbin'])
            self.fq_result = fq
            self.atoms = atoms
            return fq

    def get_pdf(self, atoms):
        """
        Calculate the atomic pair distribution factor, PDF, G(r)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate the PDF
        Returns
        -------
        1darray:
            The PDF
        """
        state = self.check_state(atoms)
        if state == [] and self.pdf_result is not None:
            if self.verbose:
                print 'using previous pdf'
            return self.pdf_result
        elif state == [] and self.fq_result is not None:
            if self.verbose:
                print 'using previous fq'
            fq = self.fq_result
        else:
            if any(x in ['exp', 'numbers'] for x in state):
                if self.verbose:
                    print 'calculating new scatter factors'
                wrap_atoms(atoms, self.exp)
                # Changing/Updating the scatter factors invalidates saved data
                self.fq_result = None
                self.pdf_result = None
                self.fq_grad_result = None
                self.pdf_grad_result = None
            fq = self.fq(atoms, self.pdf_qbin, 'PDF')
            self.fq_result = fq
        r = self.get_r()
        pdf0 = get_pdf_at_qmin(
            fq,
            self.exp['rstep'],
            self.pdf_qbin,
            r,
            self.exp['qmin']
        )
        self.pdf_result = pdf0
        self.atoms = atoms
        return pdf0

    def get_sq(self, atoms):
        """
        Calculate the structure factor S(Q)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate S(Q)
        Returns
        -------
        1darray:
            The structure factor
        """
        fq = self.get_fq(atoms)
        scatter_vector = np.arange(0, self.exp['qmax'], self.exp['qbin'])
        old_settings = np.seterr(all='ignore')
        sq = (fq / scatter_vector) + np.ones(scatter_vector.shape)
        np.seterr(**old_settings)
        sq[np.isinf(sq)] = 0.
        return sq

    def get_iq(self, atoms):
        """
        Calculate the scattering intensity, I(Q)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate I(Q)
        Returns
        -------
        1darray:
            The scattering intensity
        """
        # FIXME: there is a problem in the normalization
        return self.get_sq(atoms) * \
               np.average(atoms.get_array('F(Q) scatter')) ** 2

    def get_2d_scatter(self, atoms, pixel_array):
        """
        Calculate the scattering intensity as projected onto a detector

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate I(Q)
        pixel_array: 2darray
            A map from Q to the xy coordinates of the detector, each element
            has a Q value
        Returns
        -------
        2darray:
            The scattering intensity on the detector
        """

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
        """
        Calculate the gradient of the reduced structure factor F(Q)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate grad F(Q)
        Returns
        -------
        3darray:
            The gradient of the reduced structure factor
        """
        state = self.check_state(atoms)
        if state == [] and self.fq_grad_result is not None:
            if self.verbose:
                print 'using previous fq'
            return self.fq_grad_result
        else:
            if any(x in ['exp', 'numbers'] for x in state):
                if self.verbose:
                    print 'calculating new scatter factors'
                wrap_atoms(atoms, self.exp)
                # Changing/Updating the scatter factors invalidates saved data
                self.fq_result = None
                self.pdf_result = None
                self.fq_grad_result = None
                self.pdf_grad_result = None
        g = self.grad(atoms, self.exp['qbin'])
        self.fq_grad_result = g
        return g

    def get_grad_pdf(self, atoms):
        """
        Calculate the gradient of the PDF

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate grad PDF
        Returns
        -------
        3darray:
            The gradient of the PDF
        """
        state = self.check_state(atoms)
        if state == [] and self.pdf_grad_result is not None:
            if self.verbose:
                print 'using previous pdf'
            return self.pdf_result
        elif state and self.fq_grad_result is not None:
            if self.verbose:
                print 'using previous fq'
            fq_grad = self.fq_result
        else:
            if any(x in ['exp', 'numbers'] for x in state):
                if self.verbose:
                    print 'calculating new scatter factors'
                wrap_atoms(atoms, self.exp)
                # Changing/Updating the scatter factors invalidates saved data
                self.fq_result = None
                self.pdf_result = None
                self.fq_grad_result = None
                self.pdf_grad_result = None
            fq_grad = self.grad(atoms, self.pdf_qbin, 'PDF')
        qmin_bin = int(self.exp['qmin'] / self.pdf_qbin)
        fq_grad[:, :, :qmin_bin] = 0.
        rgrid = self.get_r()

        pdf_grad = self.grad_pdf(fq_grad, self.exp['rstep'], self.pdf_qbin,
                                 rgrid,
                                 self.exp['qmin'])
        self.pdf_grad_result = pdf_grad
        return pdf_grad

    def get_fq_voxels(self, atoms, new_atom, resolution):
        if self.check_state(atoms):
            wrap_atoms(atoms, self.exp)
        if type(resolution) == float:
            resolution = np.ones(3, np.float32) * resolution
        elif not isinstance(resolution, np.ndarray):
            raise NotImplementedError
        fq = self.fq(atoms, self.exp['qbin'], normalization=False)
        voxels = self.voxel_fq(atoms, new_atom, resolution, fq, self.exp['qbin'])
        return voxels

    def get_pdf_voxels(self, atoms, new_atom, resolution):
        if self.check_state(atoms):
            wrap_atoms(atoms, self.exp)
        r = self.get_r()
        fq = self.fq(atoms, self.pdf_qbin, 'PDF', normalization=False)
        if type(resolution) == float:
            resolution = np.ones(3, np.float32) * resolution
        elif not isinstance(resolution, np.ndarray):
            raise NotImplementedError
        voxels = self.voxel_fq(atoms, new_atom, resolution, fq, self.pdf_qbin, 'PDF')
        qmin_bin = int(self.exp['qmin'] / self.pdf_qbin)
        voxels[:, :, :, :qmin_bin] = 0.
        vpdf = self.voxel_pdf(voxels, self.exp['rstep'], self.pdf_qbin,
                                 r,
                                 self.exp['qmin'])
        return vpdf
