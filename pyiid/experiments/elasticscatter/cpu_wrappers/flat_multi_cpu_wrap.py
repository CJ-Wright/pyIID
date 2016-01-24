from multiprocessing import Pool, cpu_count
import psutil
from pyiid.experiments.elasticscatter.atomics.cpu_atomics import *

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


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
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

    Returns
    -------
    fq:1darray
        The reduced structure function
    """
    q, adps, n, qmax_bin, scatter_array = setup_cpu_calc(atoms, sum_type)
    k_max = int((n ** 2 - n) / 2.)
    allocation = cpu_k_space_fq_allocation

    master_task = [q, adps, scatter_array, qbin]

    ans = cpu_multiprocessing(atomic_fq, allocation, master_task,
                              (n, qmax_bin))

    # sum the answers
    final = np.sum(ans, axis=0, dtype=np.float64)
    final = final.astype(np.float32)
    norm = np.empty((k_max, qmax_bin), np.float32)
    get_normalization_array(norm, scatter_array, 0)
    na = np.mean(norm, axis=0, dtype=np.float32) * np.float32(n)
    # na = np.mean(norm, axis=0, dtype=np.float64) * n
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
    ans = cpu_multiprocessing(atomic_grad_fq, allocation, master_task,
                              (n, qmax_bin))
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


def cpu_multiprocessing(atomic_function, allocation,
                        master_task, constants):
    # print atomic_function, allocation, master_task, constants
    n, qmax_bin = constants
    k_max = int((n ** 2 - n) / 2.)
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
        # print type(allocation)
        m = allocation(n, qmax_bin, float(
            psutil.virtual_memory().available) / pool_size)
        if m > k_max - k_cov:
            m = k_max - k_cov
        # print m, k_cov
        sub_task = tuple(master_task + [m, k_cov])
        tasks.append(sub_task)
        k_cov += m
    # multiprocessing map problem
    # print k_cov
    ans = p.map(atomic_function, tasks)
    p.close()
    # print ans
    return ans


if __name__ == '__main__':
    from ase.atoms import Atoms
    from pyiid.experiments.elasticscatter import wrap_atoms
    # from pyiid.experiments.cpu_wrappers.nxn_cpu_wrap import wrap_fq_grad as
    #  mfqg
    import matplotlib.pyplot as plt

    plt.ion()
    n = 5000
    pos = np.random.random((n, 3)) * 10.
    atoms = Atoms('Au' + str(n), pos)
    # atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    wrap_atoms(atoms)

    # fq = wrap_fq(atoms, atoms.info['exp']['qbin'])
    # fq2 = mfq(atoms, atoms.info['exp']['qbin'])
    # print fq2.shape
    # plt.plot(fq)
    # plt.plot(fq2)
    # plt.plot((fq-fq2)/fq)
    # plt.show()
    # assert_allclose(fq, fq2, 3e-4)
    grad_fq = wrap_fq_grad(atoms, atoms.info['exp']['qbin'])
    print grad_fq
    # mgrad_fq = mfqg(atoms, atoms.info['exp']['qbin'])
    # assert_allclose(grad_fq, mgrad_fq)
    # raw_input()
