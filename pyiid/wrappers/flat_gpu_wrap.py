from pyiid.wrappers import *
from threading import Thread
from pyiid.wrappers.k_atomic_gpu import *
__author__ = 'christopher'


def subs_fq(gpu, q, scatter_array, final, qbin, m, k_cov):
    # set up GPU
    with gpu:
        final += atomic_fq(q, scatter_array, qbin, m, k_cov)


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
    q, n, qmax_bin, scatter_array, gpus, mem_list = setup_gpu_calc(atoms,
                                                                   sum_type)

    k_max = int((n ** 2 - n) / 2.)

    k_cov = 0
    p_dict = {}
    final = np.zeros(qmax_bin)

    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                m = atoms_pdf_gpu_fq(n, qmax_bin, mem)
                if m > k_max - k_cov:
                    m = k_max - k_cov
                if k_cov >= k_max:
                    break
                p = Thread(target=subs_fq, args=(
                    gpu, q, scatter_array, final, qbin, m, k_cov
                ))
                p.start()
                p_dict[gpu] = p
                k_cov += m

                if k_cov >= k_max:
                    break
    for value in p_dict.values():
        value.join()
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    final = np.nan_to_num(1 / na * final)
    np.seterr(**old_settings)
    return 2 * final


def subs_grad_fq(gpu, q, scatter_array, grad_p, qbin, k_cov, m):
    with gpu:
        grad_p += atomic_grad_fq(q, scatter_array, qbin, k_cov, m)


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    q, n, qmax_bin, scatter_array, sort_gpus, sort_gmem = setup_gpu_calc(atoms,
                                                                         sum_type)

    # setup test_flat map
    k_max = int((n ** 2 - n) / 2.)

    gpus, mem_list = get_gpus_mem()
    k_cov = 0
    p_dict = {}
    grad_p = np.zeros((n, 3, qmax_bin))
    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                m = atoms_per_gpu_grad_fq(n, qmax_bin, mem)
                if m > k_max - k_cov:
                    m = k_max - k_cov

                p = Thread(target=subs_grad_fq, args=(
                    gpu, q, scatter_array, grad_p, qbin, k_cov, m,
                ))
                p.start()
                p_dict[gpu] = p
                k_cov += m
                if k_cov >= k_max:
                    break
    for value in p_dict.values():
        value.join()
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            grad_p[tx, tz, :] = np.nan_to_num(1 / na * grad_p[tx, tz, :])
    np.seterr(**old_settings)
    return grad_p
