"""
The main class in this module `ElasticScatter` holds the experimental details,
and processor information needed to calculate the elastic powder scattering
from a collection of atoms.

Notes
-----
There are 4 layers of code from top to bottom.
1. This code, the object layer. This layer is the main point of user
interaction and keeps track of the experimental information and which
algorithm/processor to use.
2. The wrappers, these functions properly dispatch the data to processors
and atomic functions
3. The atomics, which operate on a chunk of data, they could operate on a
single piece of data and can't be interrupted/subdivided without loosing
functionality.
4. The kernels, which are compiled function run inside the atomics and perform
the actual number crunching.
"""
import math
import numpy as np
from accelerate.cuda import cuda

from pyiid.experiments.elasticscatter.cpu_wrappers.nxn_cpu_wrap import (
    wrap_fq_grad as cpu_wrap_fq_grad,
    wrap_fq as cpu_wrap_fq,
    wrap_fq_dadp as cpu_wrap_dadp)
from pyiid.experiments.elasticscatter.kernels.master_kernel import (
    grad_pdf as cpu_grad_pdf,
    get_pdf_at_qmin,
    get_scatter_array)

from scipy.interpolate import griddata
from pyiid.adp import has_adp

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
    except cuda.CudaSupportError:
        tf = False
    else:
        tf = True
    return tf


def check_cudafft():
    try:
        import accelerate.cuda.fft
        tf = True
    except ImportError:
        tf = False
        print('no cudafft')
    return tf


class ElasticScatter(object):
    """
    Scatter contains all the methods associated with producing theoretical
    diffraction patterns and PDFs from atomic configurations.  It does not
    include potential energies, such as Rw and chi**2 which are under the
    Calculator object.
    >>>from ase.atoms import Atoms
    >>>import matplotlib.pyplot as plt
    >>>atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    >>>s = ElasticScatter({'rmax': 5., 'rmin': 2.})
    >>>pdf = s.get_pdf(atoms)
    >>>atoms.rattle(.1)
    >>>pdf2 = s.get_pdf(atoms)
    >>>plt.plot(s.get_r(), pdf)
    >>>plt.plot(s.get_r(), pdf2)
    >>>plt.show()
    """

    def __init__(self, exp_dict=None, verbose=False):
        # TODO: splay out exp_dict
        self.verbose = verbose
        self.wrap_atoms_state = None

        # Currently supported processor architectures, in order of most
        # advanced to least
        self.avail_pro = ['MPI-GPU', 'GPU', 'CPU']

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
        self.adp_grad = cpu_wrap_dadp
        self.processor = 'CPU'
        self.alg = 'nxn'

        # Get the fastest processor architecture available
        self.set_processor()

    def _wrap_atoms(self, atoms):
        """
        Call this function before applying calculator, it will generate static
        arrays for the scattering, preventing recalculation

        Parameters
        -----------
        atoms: ase.Atoms
            The atoms to which scatter factors are added
        """
        if 'qbin' not in self.exp.keys():
            self.exp['qbin'] = .1
        n = len(atoms)
        e_num = atoms.get_atomic_numbers()
        e_set = set(e_num)
        e_list = list(e_set)

        for qbin, name in zip(
                [self.exp['qbin'],
                 self.pdf_qbin],
                ['F(Q) scatter', 'PDF scatter']
        ):
            qmax_bin = int(math.floor(self.exp['qmax'] / qbin))
            set_scatter_array = np.zeros((len(e_set), qmax_bin),
                                         dtype=np.float32)

            # Calculate the element-wise scatter factor array
            get_scatter_array(set_scatter_array, e_num, self.exp['qbin'])
            scatter_array = np.zeros((n, qmax_bin), dtype=np.float32)

            # Disseminate the element wise scatter factors
            for i in range(len(e_set)):
                scatter_array[np.where(atoms.numbers == e_list[i])[0], :] = \
                    set_scatter_array[i, :]

            # Set the new scatter factor array
            if name in atoms.arrays.keys():
                del atoms.arrays[name]
            atoms.set_array(name, scatter_array)

        atoms.info['exp'] = self.exp
        atoms.info['scatter_atoms'] = n

    def _check_wrap_atoms_state(self, atoms):
        """
        Check if we need to recalculate the atomic scatter factors
        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration

        Returns
        -------

        """
        t_value = True
        if self.wrap_atoms_state is None:
            t_value = False
        elif 'F(Q) scatter' not in atoms.arrays.keys():
            t_value = False
        elif (atoms.info['exp'] != self.exp
              or atoms.info['scatter_atoms'] != len(atoms)):
            t_value = False
        if not t_value:
            if self.verbose:
                print('calculating new scatter factors')
            self._wrap_atoms(atoms)
            self.wrap_atoms_state = atoms
        return t_value

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
        if processor is None:
            # Test each processor in order of most advanced to least
            for pro in self.avail_pro:
                if self.set_processor(
                        processor=pro, kernel_type=kernel_type) is not None:
                    break

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

        elif processor == self.avail_pro[1] and check_gpu() is True:
            from pyiid.experiments.elasticscatter\
                .gpu_wrappers.gpu_wrap import (wrap_fq as flat_fq,
                                               wrap_fq_grad as flat_grad)

            self.fq = flat_fq
            self.grad = flat_grad
            self.alg = 'flat'
            if check_cudafft():
                from pyiid.experiments.elasticscatter.gpu_wrappers.gpu_wrap \
                    import \
                    grad_pdf
                self.grad_pdf = grad_pdf
            else:
                self.grad_pdf = cpu_grad_pdf
            self.processor = processor
            return True

        elif processor == self.avail_pro[2]:
            if kernel_type == 'nxn':
                self.fq = cpu_wrap_fq
                self.grad = cpu_wrap_fq_grad
                self.adp_grad = cpu_wrap_dadp
                self.alg = 'nxn'

            elif kernel_type == 'flat':
                from pyiid.experiments.elasticscatter.cpu_wrappers \
                    .flat_multi_cpu_wrap import \
                    wrap_fq, wrap_fq_grad, wrap_fq_dadp

                self.fq = wrap_fq
                self.grad = wrap_fq_grad
                self.adp_grad = wrap_fq_dadp
                self.alg = 'flat'

            elif kernel_type == 'flat-serial':
                from pyiid.experiments.elasticscatter.cpu_wrappers \
                    .flat_serial_cpu_wrap import \
                    wrap_fq, wrap_fq_grad

                self.fq = wrap_fq
                self.grad = wrap_fq_grad
                self.alg = 'flat-serial'

            self.grad_pdf = cpu_grad_pdf
            self.processor = processor
            return True

    def get_fq(self, atoms, noise=None, noise_distribution=np.random.normal):
        """
        Calculate the reduced structure factor F(Q)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate F(Q)
        noise: {None, float, ndarray}, optional
            Add noise to the data, if `noise` is a float then assume flat
            gaussian noise with a standard deviation of noise, if an array
            then assume that each point has a gaussian distribution of noise
            with a standard deviation given by noise. Note that this noise is
            noise in I(Q) which is propagated to F(Q)
        noise_distribution: distribution function
            The distribution function to take the scattering pattern

        Returns
        -------
        np.ndarray:
            The reduced structure factor
        """
        self._check_wrap_atoms_state(atoms)
        fq = self.fq(atoms, self.exp['qbin'])
        fq = fq[int(np.floor(self.exp['qmin'] / self.exp['qbin'])):]
        if noise is not None and noise > 0.0:
            fq_noise = noise * np.abs(self.get_scatter_vector()) / np.abs(
                np.average(atoms.get_array('F(Q) scatter'), axis=0)
                ** 2)[int(np.floor(self.exp['qmin'] / self.exp['qbin'])):]
            if fq_noise[0] == 0.0:
                fq_noise[0] += 1e-9  # added because we can't have zero noise
            fq = noise_distribution(fq, fq_noise)
        return fq

    def get_pdf(self, atoms, noise=None, noise_distribution=np.random.normal):
        """
        Calculate the atomic pair distribution factor, PDF, G(r)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate the PDF
        noise: {None, float, np.ndarray}, optional
            Add noise to the data, if `noise` is a float then assume flat
            gaussian noise with a standard deviation of noise, if an array
            then assume that each point has a gaussian distribution of noise
            with a standard deviation given by noise. Note that this noise is
            noise in I(Q) which is propagated to F(Q)
        noise_distribution: callable
            The distribution function to take the scattering pattern

        Returns
        -------
        np.ndarray:
            The PDF
        """
        self._check_wrap_atoms_state(atoms)
        fq = self.fq(atoms, self.pdf_qbin, 'PDF')
        if noise is not None and noise > 0.0:
            a = np.abs(self.get_scatter_vector(pdf=True))
            b = np.abs(np.average(atoms.get_array('PDF scatter') ** 2, axis=0))
            if isinstance(noise, np.ndarray) and noise.shape != a.shape:
                noise = griddata(np.arange(0, noise.shape), noise, np.arange(
                    a.shape))
            fq_noise = noise * a / b
            if fq_noise[0] == 0.0:
                fq_noise[0] += 1e-9  # added because we can't have zero noise
            fq = noise_distribution(fq, fq_noise)
        r = self.get_r()
        pdf0 = get_pdf_at_qmin(
            fq,
            self.exp['rstep'],
            self.pdf_qbin,
            r,
            self.exp['qmin']
        )
        return pdf0

    def get_sq(self, atoms, noise=None, noise_distribution=np.random.normal):
        """
        Calculate the structure factor S(Q)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate S(Q)
        noise: {None, float, np.ndarray}, optional
            Add noise to the data, if `noise` is a float then assume flat
            gaussian noise with a standard deviation of noise, if an array
            then assume that each point has a gaussian distribution of noise
            with a standard deviation given by noise. Note that this noise is
            noise in I(Q) which is propagated to S(Q)
        noise_distribution: callable
            The distribution function to take the scattering pattern

        Returns
        -------
        np.ndarray:
            The structure factor
        """
        fq = self.get_fq(atoms, noise, noise_distribution)
        old_settings = np.seterr(all='ignore')
        sq = (fq / self.get_scatter_vector()) + np.ones(
            self.get_scatter_vector().shape)
        np.seterr(**old_settings)
        sq[np.isinf(sq)] = 0.
        return sq

    def get_iq(self, atoms, noise=None, noise_distribution=np.random.normal):
        """
        Calculate the scattering intensity, I(Q)

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate I(Q)
        noise: {None, float, np.ndarray}, optional
            Add noise to the data, if `noise` is a float then assume flat
            gaussian noise with a standard deviation of noise, if an array
            then assume that each point has a gaussian distribution of noise
            with a standard deviation given by noise. Note that this noise is
            noise in I(Q) which is propagated to I(Q)
        noise_distribution: callable
            The distribution function to take the scattering pattern

        Returns
        -------
        np.ndarray:
            The scattering intensity
        """
        sq = self.get_sq(atoms, noise, noise_distribution)
        f2 = np.average(atoms.get_array('F(Q) scatter'), axis=0) ** 2
        iq = sq * f2[int(np.floor(self.exp['qmin'] / self.exp['qbin'])):]
        return iq

    def get_2d_scatter(self, atoms, pixel_array):
        """
        Calculate the scattering intensity as projected onto a detector

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate I(Q)
        pixel_array: np.ndarray
            A map from Q to the xy coordinates of the detector, each element
            has a Q value
        Returns
        -------
        np.ndarray:
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
        np.ndarray:
            The gradient of the reduced structure factor
        """
        self._check_wrap_atoms_state(atoms)
        g = self.grad(atoms, self.exp['qbin'])
        return g[:, :, int(np.floor(self.exp['qmin'] / self.exp['qbin'])):]

    def get_grad_pdf(self, atoms):
        """
        Calculate the gradient of the PDF

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate grad PDF
        Returns
        -------
        np.ndarray:
            The gradient of the PDF
        """
        self._check_wrap_atoms_state(atoms)
        fq_grad = self.grad(atoms, self.pdf_qbin, 'PDF')
        qmin_bin = int(self.exp['qmin'] / self.pdf_qbin)
        fq_grad[:, :, :qmin_bin] = 0.
        rgrid = self.get_r()

        pdf_grad = self.grad_pdf(fq_grad, self.exp['rstep'], self.pdf_qbin,
                                 rgrid,
                                 self.exp['qmin'])
        return pdf_grad

    def get_scatter_vector(self, pdf=False):
        """
        Calculate the scatter vector Q for the current experiment

        Parameters
        ----------
        pdf: bool
            If true return the PDF rendering scatter vector

        Returns
        -------
        np.ndarray:
            The Q range for this experiment
        """
        if pdf:
            return np.arange(0.,
                             math.floor(self.exp['qmax'] / self.pdf_qbin) *
                             self.pdf_qbin, self.pdf_qbin)
        return np.arange(self.exp['qmin'], math.floor(self.exp['qmax'] /
                                                      self.exp['qbin']) *
                         self.exp['qbin'],
                         self.exp['qbin'])

    def get_r(self):
        """
        Calculate the inter-atomic distance range for the current experiment

        Returns
        -------
        np.ndarray:
            The r range for this experiment
        """
        return np.arange(self.exp['rmin'], self.exp['rmax'], self.exp['rstep'])

    def get_grad_adp_fq(self, atoms):
        self._check_wrap_atoms_state(atoms)
        if has_adp(atoms):
            # Calculate the grad of the adps using the wrappers
            grad_adp_fq = self.adp_grad(atoms, self.exp['qbin'])
        else:
            print("It seems that the atoms don't have any adps, "
                  "you need adps to call this method")
            grad_adp_fq = np.zeros(atoms.get_positions().shape)
        return grad_adp_fq

    def get_grad_adp_pdf(self, atoms):
        """
        Calculate the gradient of the PDF

        Parameters
        ----------
        atoms: ase.Atoms
            The atomic configuration for which to calculate grad PDF
        Returns
        -------
        np.ndarray:
            The gradient of the PDF
        """
        self._check_wrap_atoms_state(atoms)
        fq_grad = self.adp_grad(atoms, self.pdf_qbin, 'PDF')
        qmin_bin = int(self.exp['qmin'] / self.pdf_qbin)
        fq_grad[:, :, :qmin_bin] = 0.
        rgrid = self.get_r()

        pdf_grad = self.grad_pdf(fq_grad, self.exp['rstep'], self.pdf_qbin,
                                 rgrid,
                                 self.exp['qmin'])
        return pdf_grad * -1.
