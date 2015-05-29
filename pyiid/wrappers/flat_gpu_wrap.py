__author__ = 'christopher'
import numpy as np
from numba import cuda
import math
from threading import Thread
import sys

sys.path.extend(['/mnt/work-data/dev/pyIID'])

from pyiid.kernels.flat_kernel import antisymmetric_reshape, symmetric_reshape, \
    get_ij_lists


def get_gpus_mem():
    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(int(meminfo[0]))
    sort_gpus = [x for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    sort_gmem = [y for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    return sort_gpus, sort_gmem


def subs_fq(gpu, q, scatter_array, fq_q, qmax_bin, qbin, il, jl):
    # set up GPU
    with gpu:
    # load kernels
        from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
            get_normalization_array, get_fq, d2_to_d1_sum, construct_qij, \
            construct_scatij

        elements_per_dim_1 = [len(il)]
        tpb1 = [32]
        bpg1 = []
        for e_dim, tpb in zip(elements_per_dim_1, tpb1):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg1.append(bpg)

        elements_per_dim_q = [qmax_bin]
        tpbq = [32]
        bpgq = []
        for e_dim, tpb in zip(elements_per_dim_q, tpb1):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpgq.append(bpg)

        elements_per_dim_2 = [len(il), qmax_bin]
        tpb2 = [16, 4]
        bpg2 = []
        for e_dim, tpb in zip(elements_per_dim_2, tpb2):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg2.append(bpg)

        stream = cuda.stream()
        stream2 = cuda.stream()

        # transfer data
        dd = cuda.device_array((len(il), 3), dtype=np.float32, stream=stream)

        dqi = cuda.device_array((len(il), 3), dtype=np.float32, stream=stream)
        dqj = cuda.device_array((len(il), 3), dtype=np.float32, stream=stream)
        dil = cuda.to_device(np.asarray(il, dtype=np.int32), stream=stream)
        djl = cuda.to_device(np.asarray(jl, dtype=np.int32), stream=stream)
        dq = cuda.to_device(q)

        # calculate kernels
        construct_qij[bpg1, tpb1, stream](dqi, dqj, dq, dil, djl)

        dr = cuda.device_array(len(il), dtype=np.float32, stream=stream)
        get_d_array[bpg1, tpb1, stream](dd, dqi, dqj)
        del dqi, dqj

        get_r_array[bpg1, tpb1, stream](dr, dd)

        dnorm = cuda.device_array((len(il), qmax_bin), dtype=np.float32,
                                  stream=stream2)

        dscati = cuda.device_array((len(il), qmax_bin), dtype=np.float32,
                                   stream=stream2)
        dscatj = cuda.device_array((len(il), qmax_bin), dtype=np.float32,
                                   stream=stream2)
        dscat = cuda.to_device(scatter_array.astype(np.float32), stream=stream2)

        construct_scatij[bpg2, tpb2, stream2](dscati, dscatj, dscat, dil, djl)

        get_normalization_array[bpg2, tpb2, stream2](dnorm, dscati, dscatj)
        del dscati, dscatj, dscat, dil, djl

        dfq = cuda.device_array((len(il), qmax_bin), dtype=np.float32,
                                stream=stream2)

        final = np.zeros(qmax_bin, dtype=np.float32)
        dfinal = cuda.to_device(final)

        get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)
        del dr, dnorm
        d2_to_d1_sum[bpgq, tpbq, stream2](dfinal, dfq)
        del dfq

        dfinal.to_host(stream2)
        fq_q.append(final)
        del dfinal


def wrap_fq(atoms, qmax=25., qbin=.1):
    # set up atoms
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(math.ceil(qmax / qbin))
    scatter_array = atoms.get_array('scatter')

    # setup flat map
    il, jl = get_ij_lists(n)
    k_max = len(il)

    gpus, mem_list = get_gpus_mem()

    fq_q = []
    k_cov = 0
    p_dict = {}

    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            m = int(math.floor(
                    float(mem-4*qmax_bin-4*qmax_bin*n-12*n) / (16*(qmax_bin + 3))))

            if m > k_max - k_cov:
                m = k_max - k_cov
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                if k_cov >= k_max:
                    break
                p = Thread(target=subs_fq, args=(
                    gpu, q, scatter_array, fq_q, qmax_bin, qbin,
                    il[k_cov:k_cov+m],
                    jl[k_cov:k_cov+m],
                ))
                p.start()
                p_dict[gpu] = p
                k_cov += m

                if k_cov >= k_max:
                    break
    for value in p_dict.values():
        value.join()
    final = np.zeros(qmax_bin)
    for ele in fq_q:
        final[:] += ele
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    final = np.nan_to_num(1 / na * final)
    np.seterr(**old_settings)
    return 2 * final


def subs_grad_fq(gpu, q, scatter_array, grad_q, qmax_bin, qbin, il, jl, k_cov, index_list):
    # set up GPU
    n = len(q)
    with gpu:
        # load kernels
        from pyiid.kernels.flat_kernel import get_d_array, get_r_array, \
            get_normalization_array, get_fq, d2_to_d1_sum, get_grad_fq, \
            construct_scatij, construct_qij, flat_sum

        elements_per_dim_1 = [len(il)]
        tpb1 = [32]
        bpg1 = []
        for e_dim, tpb in zip(elements_per_dim_1, tpb1):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg1.append(bpg)

        elements_per_dim_q = [qmax_bin]
        tpbq = [32]
        bpgq = []
        for e_dim, tpb in zip(elements_per_dim_q, tpb1):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpgq.append(bpg)

        elements_per_dim_2 = [len(il), qmax_bin]
        tpb2 = [16, 4]
        bpg2 = []
        for e_dim, tpb in zip(elements_per_dim_2, tpb2):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg2.append(bpg)


        elements_per_dim_nq = [len(q), qmax_bin]
        tpbnq = [16, 4]
        bpgnq = []
        for e_dim, tpb in zip(elements_per_dim_nq, tpbnq):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpgnq.append(bpg)

        elements_per_dim_3 = [len(il), len(q), qmax_bin]
        tpb3 = [64, 1, 1]
        bpg3 = []
        for e_dim, tpb in zip(elements_per_dim_3, tpb3):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg3.append(bpg)

        stream = cuda.stream()
        stream2 = cuda.stream()

        # transfer data
        dd = cuda.device_array((len(il), 3), dtype=np.float32, stream=stream)

        dq = cuda.to_device(q)
        dqi = cuda.device_array((len(il), 3), dtype=np.float32, stream=stream)
        dqj = cuda.device_array((len(il), 3), dtype=np.float32, stream=stream)
        dil = cuda.to_device(np.asarray(il, dtype=np.int32), stream=stream)
        djl = cuda.to_device(np.asarray(jl, dtype=np.int32), stream=stream)

        construct_qij[bpg1, tpb1, stream](dqi, dqj, dq, dil, djl)
        dr = cuda.device_array(len(il), dtype=np.float32, stream=stream)

        # calculate kernels
        get_d_array[bpg1, tpb1, stream](dd, dqi, dqj)
        del dqi, dqj

        get_r_array[bpg1, tpb1, stream](dr, dd)

        dnorm = cuda.device_array((len(il), qmax_bin), dtype=np.float32,
                                  stream=stream2)
        dscati = cuda.device_array((len(il), qmax_bin), dtype=np.float32,
                                   stream=stream2)
        dscatj = cuda.device_array((len(il), qmax_bin), dtype=np.float32,
                                   stream=stream2)
        dscat = cuda.to_device(scatter_array.astype(np.float32), stream=stream2)

        construct_scatij[bpg2, tpb2, stream2](dscati, dscatj, dscat, dil, djl)
        get_normalization_array[bpg2, tpb2, stream2](dnorm, dscati, dscatj)
        del dscati, dscatj, dscat

        dfq = cuda.device_array((len(il), qmax_bin), dtype=np.float32,
                                stream=stream2)

        get_fq[bpg2, tpb2, stream2](dfq, dr, dnorm, qbin)

        grad = np.zeros((len(il), 3, qmax_bin), dtype=np.float32)
        dgrad = cuda.device_array(grad.shape, dtype=np.float32, stream=stream2)


        get_grad_fq[bpg2, tpb2, stream2](dgrad, dfq, dr, dd, dnorm, qbin)

        new_grad2 = np.zeros((len(q), 3, qmax_bin), dtype=np.float32)
        dnew_grad = cuda.to_device(new_grad2, stream=stream2)
        # dnew_grad = cuda.device_array(new_grad2.shape, dtype=np.float32, stream=stream2)

        # jil = np.zeros((len(q), len(q)-1), dtype=np.int32)
        print min(il), min(jl), max(il), max(jl)
        print '\n\n\n'
        min_n = min([np.min(il), np.min(jl)])

        jil = np.zeros((n, n-1), dtype=np.int32)

        for sn in range(n):
            jil[sn, :] = np.concatenate([np.where(jl == sn)[0], np.where(il == sn)[0]])
        print jil
        djil = cuda.to_device(jil, stream2)
        flat_sum[bpgnq, tpbnq, stream2](dnew_grad, dgrad, djil)
        dnew_grad.to_host(stream2)
        del dd, dr, dnorm, dfq, dgrad, dil, djl
    grad_q.append(new_grad2)
    index_list.append(k_cov)
    del grad, k_cov, il, jl


def wrap_fq_grad(atoms, qmax=25, qbin=.1):
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(math.ceil(qmax / qbin))
    scatter_array = atoms.get_array('scatter')

    # setup flat map

    il, jl = get_ij_lists(n)
    # print il, jl
    k_max = len(il)

    gpus, mem_list = get_gpus_mem()
    grad_q = []
    index_list = []
    k_cov = 0
    p_dict = {}

    while k_cov < k_max:
        for gpu, mem in zip(gpus, mem_list):
            m = int(math.floor(
                float(mem - 4*qmax_bin*n - 12*n)/(4*(7*qmax_bin + 12))))
            # m = 2
            if m > k_max - k_cov:
                m = k_max - k_cov
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                # if k_cov >= k_max:
                #     break
                # print float(k_cov+m)/k_max * 100, '%'
                p = Thread(target=subs_grad_fq, args=(
                    gpu, q, scatter_array, grad_q, qmax_bin, qbin,
                    il[k_cov:k_cov+m],
                    jl[k_cov:k_cov+m],
                    k_cov,
                    index_list,
                ))
                p.start()
                p_dict[gpu] = p
                k_cov += m

                if k_cov >= k_max:
                    break
        #TODO: sum arrays during processing to cut down on memory
    for value in p_dict.values():
        value.join()

    sort_grads = [x for (y, x) in sorted(zip(index_list, grad_q))]

    if len(sort_grads) > 1:
        # grads = np.concatenate(sort_grads, axis=)
        grad_p = np.sum(sort_grads, axis=0)
        print grad_p.shape
    else:
        grad_p = sort_grads[0]
    '''
    jil = np.zeros((len(q), len(q)-1), dtype=np.int32)
    for n in range(len(q)):
        jil[n, :] = np.concatenate([np.where(jl == n)[0], np.where(il == n)[0]])
    grad_p = np.zeros((len(q), 3, qmax_bin))
    print jil
    for n in range(len(q)):
        grad_p[n, :, :] += grads[jil[n, :n], :, :].sum(axis=0)
        grad_p[n, :, :] -= grads[jil[n, n:], :, :].sum(axis=0)
    '''
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            grad_p[tx, tz, :] = np.nan_to_num(1 / na * grad_p[tx, tz, :])
    np.seterr(**old_settings)
    return grad_p


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('''
    from ase.atoms import Atoms
    import os
    from pyiid.wrappers.scatter import wrap_atoms
    import matplotlib.pyplot as plt

    # n = 300
    # pos = np.random.random((n, 3)) * 10.
    # atoms = Atoms('Au' + str(n), pos)
    atoms = Atoms('Au4', [[0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0]])
    wrap_atoms(atoms)

    fq = wrap_fq(atoms)
    grad_fq = wrap_fq_grad(atoms)
    # print grad_fq