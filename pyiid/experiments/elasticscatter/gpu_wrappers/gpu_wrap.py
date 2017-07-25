from threading import Thread
from accelerate.cuda import fft
from pyiid.experiments.elasticscatter.atomics.gpu_atomics import *
from pyiid.experiments import *
from pyiid.experiments.elasticscatter.kernels.cpu_flat import \
    get_normalization_array
from pyiid.adp import has_adp
from builtins import range

__author__ = 'christopher'


def setup_gpu_calc(atoms, sum_type):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter').astype(np.float32)
    else:
        scatter_array = atoms.get_array('PDF scatter').astype(np.float32)
    qmax_bin = scatter_array.shape[1]
    sort_gpus, sort_gmem = get_gpus_mem()

    adps = has_adp(atoms)
    if adps:
        return q, adps.get_positions().astype(
            np.float32), n, qmax_bin, scatter_array, sort_gpus, sort_gmem
    return q, None, n, qmax_bin, scatter_array, sort_gpus, sort_gmem


def subs_fq(fq, q, adps, scatter_array, qbin, gpu, k_cov, k_per_thread):
    """
    Thread function to calculate a chunk of F(Q)

    Parameters
    ----------
    gpu: numba GPU context
        The GPU on which to run
    q: Nx3 array
        Atomic positions
    scatter_array: NxQ array
        Atomic scatter factors
    fq: np.ndarray
        The F(Q), note that we add to it which prevents a rather nasty
        memory leak
    qbin: float
        The Q resolution in A**-1
    k_cov: int
        Number of atomic pairs previously covered
    k_per_thread: int
        Number of atomic pairs to cover in this instance
    """
    # set up GPU
    with gpu:
        fq += atomic_fq(q, adps, scatter_array, qbin, k_cov, k_per_thread)


def subs_grad_fq(grad_p, q, adps, scatter_array, qbin, gpu, k_cov,
                 k_per_thread):
    """
    Thread function to calculate a chunk of F(Q)

    Parameters
    ----------
    gpu: numba GPU context
        The GPU on which to run
    q: Nx3 array
        Atomic positions
    scatter_array: NxQ array
        Atomic scatter factors
    grad_p: 3darray
        The grad F(Q), note that we add to it which prevents a rather nasty
        memory leak
    qbin: float
        The Q resolution in A**-1
    k_cov: int
        Number of atomic pairs previously covered
    k_per_thread: int
        Number of atomic pairs to cover in this instance
    """
    # set up GPU
    with gpu:
        grad_p += atomic_grad_fq(q, adps, scatter_array, qbin, k_cov,
                                 k_per_thread)


def sub_grad_pdf(gpu, gpadc, gpadcfft, atoms_per_thread, n_cov):
    """
    Thread function to calculate a chunk of grad PDF

    Parameters
    ----------
    gpu: numba GPU context
        The GPU to use
    gpadc: Nx3xQ array
        The input grad F(Q) array, padded and symmetry corrected
    gpadcfft: Nx3xr array
        The output grad PDF array, needs to be post processed
    atoms_per_thread: int
        Number of atoms being processed in this thread
    n_cov: int
        Number of atoms previously covered
    """
    input_shape = tuple([gpadcfft.shape[-1]])
    with gpu:
        batch_operations = atoms_per_thread
        plan = fft.FFTPlan(input_shape, np.complex64, np.complex64,
                           batch_operations)
        for i in range(3):
            batch_input = np.ravel(
                gpadc[n_cov:n_cov + atoms_per_thread, i, :]).astype(
                np.complex64)
            batch_output = np.zeros(batch_input.shape, dtype=np.complex64)
            _ = plan.inverse(batch_input, out=batch_output)
            del batch_input
            data_out = np.reshape(batch_output,
                                  (atoms_per_thread, input_shape[0]))
            data_out /= input_shape[0]

            gpadcfft[n_cov:n_cov + atoms_per_thread, i, :] = data_out
            del data_out, batch_output


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
    """
    Function which handles all the threads for the computation of F(Q)

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qbin: float
        The Q resolution in A**-1
    sum_type: str
        The type of calculation being run, defaults to F(Q), but runs PDF
        if anything other than 'fq' is provided.  This is needed because
        the PDF runs at a different Q resolution and thus a different scatter
        factor array for the atoms
    Returns
    -------
    np.ndarray;
        The reduced structure factor
    """
    q, adps, n, qmax_bin, scatter_array, gpus, mem_list = setup_gpu_calc(
        atoms, sum_type)
    allocation = gpu_k_space_fq_allocation

    fq = np.zeros(qmax_bin, np.float64)
    master_task = [fq, q, adps, scatter_array, qbin]
    fq = gpu_multithreading(subs_fq, allocation, master_task, (n, qmax_bin),
                            (gpus, mem_list))
    fq = fq.astype(np.float32)
    norm = np.empty((int(n * (n - 1) / 2.), qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, 0)
    na = np.mean(norm, axis=0) * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(fq / na)
    np.seterr(**old_settings)
    # Note we only calculated half of the scattering, but the symmetry allows
    # us to multiply by 2 and get it correct
    return 2 * fq


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    """
    Function which handles all the threads for the computation of grad F(Q)

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qbin: float
        The Q resolution in A**-1
    sum_type: str
        The type of calculation being run, defaults to F(Q), but runs PDF
        if anything other than 'fq' is provided.  This is needed because
        the PDF runs at a different Q resolution and thus a different scatter
        factor array for the atoms
    Returns
    -------
    np.ndarray;
        The reduced structure factor
    """
    q, adps, n, qmax_bin, scatter_array, gpus, mem_list = setup_gpu_calc(
        atoms, sum_type)
    allocation = gpu_k_space_grad_fq_allocation

    grad_p = np.zeros((n, 3, qmax_bin))
    master_task = [grad_p, q, adps, scatter_array, qbin]
    grad_p = gpu_multithreading(subs_grad_fq, allocation, master_task,
                                (n, qmax_bin),
                                (gpus, mem_list))
    norm = np.empty((int(n * (n - 1) / 2.), qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, 0)
    na = np.mean(norm, axis=0) * n
    old_settings = np.seterr(all='ignore')
    grad_p = np.nan_to_num(grad_p / na)
    np.seterr(**old_settings)
    return grad_p


def grad_pdf(grad_fq, rstep, qstep, rgrid, qmin):
    """
    Function which handles all the threads for the computation of grad PDF

    Parameters
    ----------
    grad_fq: Nx3xQ array
        The gradient of F(Q)
    rstep: float
        The r resolution in A
    qstep: float
        The Q resolution in A**-1
    rgrid: np.ndarray
        The inter-atomic distance grid to map onto
    qmin: float
        The minimum Q in the experiment in A**-1
    Returns
    -------
    ndarray
        The gradient of the PDF
    """
    # Number of atoms
    n = len(grad_fq)

    # set values below Qmin to zero
    grad_fq[:, :, :int(math.ceil(qmin / qstep))] = 0.0

    # Pad to the correct length for FFT
    nfromdr = int(math.ceil(math.pi / rstep / qstep))
    if nfromdr > int(len(grad_fq)):
        # put in a bunch of zeros
        fpad2 = np.zeros((n, 3, nfromdr))
        fpad2[:, :, :grad_fq.shape[-1]] = grad_fq
        grad_fq = fpad2

    # make the symmetry correct for the PDF FFT
    padrmin = int(round(qmin / qstep))
    npad1 = padrmin + grad_fq.shape[-1]

    npad2 = (1 << int(math.ceil(math.log(npad1, 2)))) * 2

    npad4 = 4 * npad2
    gpadc = np.zeros((n, 3, npad4))
    gpadc[:, :, :2 * grad_fq.shape[-1]:2] = grad_fq[:, :, :]
    gpadc[:, :, -2:-2 * grad_fq.shape[-1] + 1:-2] = -1 * grad_fq[:, :, 1:]
    gpadcfft = np.zeros(gpadc.shape, dtype=complex)

    gpus, mems = get_gpus_mem()
    n_cov = 0
    p_dict = {}

    # Put jobs onto GPUs while we haven't finished all the atoms
    while n_cov < n:
        for gpu, mem in zip(gpus, mems):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                # TODO: Note that the second division by a memory scale factor
                # is not needed but I don't know the official memory
                # allocation and this makes certain we don't crash the system
                # on a memory allocation, should put this in a config file
                memory_scale_factor = 1.7
                atoms_per_thread = int(
                    math.floor(.8 * mem / gpadcfft.shape[
                        -1] / 8 / 2 / memory_scale_factor))
                if atoms_per_thread > n - n_cov:
                    atoms_per_thread = n - n_cov
                if n_cov >= n:
                    break
                p = Thread(target=sub_grad_pdf,
                           args=(
                               gpu, gpadc, gpadcfft, atoms_per_thread,
                               n_cov))
                p.start()
                p_dict[gpu] = p
                n_cov += atoms_per_thread
                if n_cov >= n:
                    break
    for value in p_dict.values():
        value.join()
    # cleanup data to correct size
    g = np.zeros((n, 3, npad2), dtype=complex)
    g[:, :, :] = gpadcfft[:, :, :npad2 * 2:2] * npad2 * qstep

    gpad = g.imag * 2.0 / math.pi  # type: np.ndarray
    drpad = math.pi / (gpad.shape[-1] * qstep)
    # re-bin the data onto the correct rgrid
    pdf0 = np.zeros((n, 3, len(rgrid)))
    axdrp = rgrid / drpad / 2
    aiplo = axdrp.astype(np.int)
    aiphi = aiplo + 1
    awphi = axdrp - aiplo
    awplo = 1.0 - awphi
    pdf0[:, :, :] = awplo[:] * gpad[:, :, aiplo] + awphi * gpad[:, :, aiphi]
    pdf1 = pdf0 * 2
    return pdf1.real


def gpu_multithreading(subs_function, allocation,
                       master_task, constants, gpu_info):
    n, qmax_bin = constants
    gpus, mem_list = gpu_info
    k_max = int((n ** 2 - n) / 2.)

    k_cov = 0
    p_dict = {}

    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                k_per_thread = allocation(n, qmax_bin, mem)
                if k_per_thread > k_max - k_cov:
                    k_per_thread = k_max - k_cov
                if k_cov >= k_max:
                    break
                p = Thread(target=subs_function, args=(
                    tuple(master_task + [gpu, k_cov, k_per_thread])))
                p.start()
                p_dict[gpu] = p
                k_cov += k_per_thread

                if k_cov >= k_max:
                    break
    for value in p_dict.values():
        value.join()
    return master_task[0]
