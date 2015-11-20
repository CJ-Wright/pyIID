from multiprocessing import Pool, cpu_count
import psutil
from pyiid.experiments.elasticscatter.atomics.cpu_atomics import *
from pyiid.experiments.elasticscatter.kernels.master_kernel import \
    get_single_scatter_array

__author__ = 'christopher'


def setup_cpu_calc(atoms, sum_type):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter').astype(np.float32)
    else:
        scatter_array = atoms.get_array('PDF scatter').astype(np.float32)
    n, qmax_bin = scatter_array.shape
    return q.astype(np.float32), None, n, qmax_bin, scatter_array.astype(
        np.float32)


def wrap_fq(atoms, qbin=.1, sum_type='fq', normalization=True):
    """
    Generate the reduced structure function

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qbin: float
        The size of the scatter vector increment
    sum_type: {'fq', 'pdf'}
        Which scatter array should be used for the calculation
    normalization: bool
        If True apply normalization, else don't
    Returns
    -------
    fq:1darray
        The reduced structure function
    """
    q, adps, n, qmax_bin, scatter_array = setup_cpu_calc(atoms, sum_type)
    k_max = int((n ** 2 - n) / 2.)
    allocation = cpu_k_space_fq_allocation

    master_task = [q, adps, scatter_array, qbin]

    ans = cpu_multiprocessing(atomic_fq, allocation, (n, qmax_bin),
                              master_task, k_max)

    # sum the answers
    final = np.sum(ans, axis=0, dtype=np.float64)
    final = final.astype(np.float32)
    if normalization:
        norm = np.empty((k_max, qmax_bin), np.float32)
        get_normalization_array(norm, scatter_array, 0)
        na = np.mean(norm, axis=0, dtype=np.float32) * np.float32(n)
        old_settings = np.seterr(all='ignore')
        final = np.nan_to_num(final / na)
        np.seterr(**old_settings)
    del q, n, qmax_bin, scatter_array, k_max, ans
    return 2 * final


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    """
    Generate the reduced structure function gradient

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qbin: float
        The size of the scatter vector increment
    sum_type: {'fq', 'pdf'}
        Which scatter array should be used for the calculation

    Returns
    -------

    dfq_dq:ndarray
        The reduced structure function gradient
    """
    # setup variables of interest
    q, adps, n, qmax_bin, scatter_array = setup_cpu_calc(atoms, sum_type)
    k_max = int((n ** 2 - n) / 2.)
    if k_max == 0:
        return np.zeros((n, 3, qmax_bin)).astype(np.float32)
    allocation = k_space_grad_fq_allocation
    master_task = [q, adps, scatter_array, qbin]
    ans = cpu_multiprocessing(atomic_grad_fq, allocation, (n, qmax_bin),
                              master_task, k_max)
    # sum the answers
    # print ans
    grad_p = np.sum(ans, axis=0)
    # print grad_p.shape
    norm = np.empty((k_max, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, 0)
    na = np.mean(norm, axis=0) * n
    old_settings = np.seterr(all='ignore')
    grad_p = np.nan_to_num(grad_p / na)
    np.seterr(**old_settings)
    del q, n, qmax_bin, scatter_array, k_max, ans
    return grad_p


def wrap_voxel_fq(atoms, new_atom, resolution, fq, qbin=.1, sum_type='fq'):
    """
    Generate the reduced structure function

    Parameters
    ----------
    atoms: ase.Atoms
        The atomic configuration
    qbin: float
        The size of the scatter vector increment
    sum_type: 'fq' or 'pdf'
        Determines the type of scatter array to use
    Returns
    -------
    fq:1darray
        The reduced structure function
    """
    # Pre-pool
    # get scatter array
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    new_scatter = np.zeros(scatter_array.shape[1], np.float32)
    get_single_scatter_array(new_scatter, new_atom.number, qbin)

    # define scatter_q information and initialize constants
    # Get normalization array
    norm = np.float32(scatter_array * new_scatter)
    # Set up all the variables of interest
    qbin = np.float32(qbin)
    q = atoms.get_positions().astype(np.float32)
    resolution = np.float32(resolution)
    v = np.int32(np.ceil(np.diagonal(atoms.get_cell()) / resolution))
    n, qmax_bin = scatter_array.shape

    norm2 = np.zeros((n * (n - 1) / 2., qmax_bin), np.float32)
    get_normalization_array(norm2, np.vstack((scatter_array, new_scatter)), 0)
    na = np.mean(norm2, axis=0, dtype=np.float32) * np.float32(n + 1)

    master_task = [q, norm, fq, na, qbin, resolution, v]
    # Inside pool
    vfq_list = cpu_multiprocessing(atomic_voxel_fq, voxel_fq_allocation,
                                   (n, qmax_bin, v), master_task,
                                   np.product(v))
    # Post-Pool
    vfq = None
    for sub_fq in vfq_list:
        if vfq is None:
            vfq = sub_fq
        else:
            vfq = np.vstack((vfq, sub_fq))
    # Normalize fq
    vfq = vfq.reshape(tuple(v) + (qmax_bin,))
    return vfq


def cpu_multiprocessing(atomic_function, allocation, allocation_args,
                        master_task, constants):
    # print atomic_function, allocation, master_task, constants
    k_max = constants
    # TODO: what if n is 1 kmax = 0???
    # break up problem
    pool_size = cpu_count()
    if pool_size <= 0:
        pool_size = 1
    p = Pool(pool_size, maxtasksperchild=1)
    tasks = []
    k_cov = 0
    # print k_max
    while k_cov < k_max:
        m = allocation(float(psutil.virtual_memory().available),
                       *allocation_args)
        if m > k_max - k_cov:
            m = k_max - k_cov
        for i in range(pool_size):
            if not k_cov < k_max:
                break
            n = math.ceil(float(m) / pool_size)
            if n > k_max - k_cov:
                n = k_max - k_cov
            sub_task = tuple(master_task + [n, k_cov])
            tasks.append(sub_task)
            k_cov += n
    # multiprocessing map problem
    # print k_cov
    ans = p.map(atomic_function, tasks)
    p.close()
    # print ans
    return ans