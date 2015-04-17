__author__ = 'christopher'
import numpy as np
from numba import cuda
import math
from threading import Thread


def sub_fq(gpu, q, scatter_array, fq_q, qmax_bin, qbin, m, n_cov):
    n = len(q)
    tups = [(m, n, 3), (m, n), (m, n, qmax_bin), (m, n, qmax_bin)]
    data = [np.zeros(shape=tup, dtype=np.float32) for tup in tups]
    # Kernel
    # cuda kernel information
    with gpu:
        from pyiid.kernels.multi_cuda import get_d_array, \
            get_normalization_array, get_r_array, get_fq_p0_1_2, get_fq_p3

        stream = cuda.stream()
        stream2 = cuda.stream()

        # two kinds of test_kernels; NxN or NxNxQ
        # NXN
        elements_per_dim_2 = [m, n]
        tpb_l_2 = [32, 32]
        bpg_l_2 = []
        for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg_l_2.append(bpg)

        # NxNxQ
        elements_per_dim_3 = [m, n, qmax_bin]
        tpb_l_3 = [16, 16, 4]
        bpg_l_3 = []
        for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
            bpg = int(math.ceil(float(e_dim) / tpb))
            if bpg < 1:
                bpg = 1
            assert (bpg * tpb >= e_dim)
            bpg_l_3.append(bpg)

        # start calculations

        dscat = cuda.to_device(scatter_array, stream2)
        dnorm = cuda.to_device(data[2], stream2)
        get_normalization_array[bpg_l_3, tpb_l_3, stream2](dnorm, dscat, n_cov)

        dd = cuda.to_device(data[0], stream)
        dq = cuda.to_device(q, stream)
        dr = cuda.to_device(data[1], stream)
        dfq = cuda.to_device(data[3], stream)

        get_d_array[bpg_l_2, tpb_l_2, stream](dd, dq, n_cov)
        get_r_array[bpg_l_2, tpb_l_2, stream](dr, dd)
        get_fq_p0_1_2[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin)
        get_fq_p3[bpg_l_3, tpb_l_3, stream](dfq, dnorm)
        dfq.to_host(stream)

        fq_q.append(data[3].sum(axis=(0, 1)))
        del data, dscat, dnorm, dd, dq, dr, dfq


def wrap_fq(atoms, qmax=25., qbin=.1):

    # get information for FQ transformation
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    # get info on our gpu setup and memory requrements
    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(meminfo[0])
    sort_gpus = [x for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    sort_gmem = [y for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    gpu_total_mem = sum(sort_gmem)
    total_req_mem = (2*qmax_bin*n*n+qmax_bin*n+4*n*n+3*n)*4

    # starting buffers
    fq_q = []
    n_cov = 0
    p_dict = {}

    # TODO: NEEDS WORK GIVES BAD RESULTS WITH EQUAL WEIGHTING
    # if total_req_mem < gpu_total_mem:
    if False:
        # Then the total gpu space is larger than our problem, we should give
        # out atoms by the size of the free memory, giving each gpu some work
        # to do based on its capacity.
        gpu_m = [int(round(n*float(g_mem)/gpu_total_mem)) for g_mem in sort_gmem]
        for gpu, m in zip(sort_gpus, gpu_m):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                if n_cov >= n:
                    break
                p = Thread(
                    target=sub_fq, args=(
                        gpu, q, scatter_array,
                        fq_q, qmax_bin, qbin, m, n_cov))
                p_dict[gpu] = p
                p.start()
                n_cov += m
                if n_cov >= n:
                    break
        assert n_cov == n
    else:
        # The total amount of work is greater than the sum of our GPUs, no
        # special distribution needed, just keep putting problems on GPUs until
        # finished.
        while n_cov < n:
            for gpu, mem in zip(sort_gpus, sort_gmem):
                m = int(math.floor(float(-4 * n * qmax_bin - 12 * n + .8 * mem) / (
                    8 * n * (qmax_bin + 2))))
                if m > n - n_cov:
                    m = n - n_cov

                if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                    if n_cov >= n:
                        break
                    p = Thread(
                        target=sub_fq, args=(
                            gpu, q, scatter_array,
                            fq_q, qmax_bin, qbin, m, n_cov))
                    p_dict[gpu] = p
                    p.start()
                    n_cov += m
                    if n_cov >= n:
                        break

    for value in p_dict.values():
        value.join()

    fq = np.zeros(qmax_bin)
    for ele in fq_q:
        fq[:] += ele
    na = np.average(scatter_array, axis=0)**2 * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / na * fq)
    np.seterr(**old_settings)
    return fq


def sub_grad(gpu, q, scatter_array, grad_q, qmax_bin, qbin, m, n_cov,
             index_list):
    # Build Data arrays
    n = len(q)
    tups = [(m, n, 3), (m, n),
            (m, n, qmax_bin), (m, n, qmax_bin),
            (m, n, 3, qmax_bin), (m, n, qmax_bin)]
    data = [np.zeros(shape=tup, dtype=np.float32) for tup in tups]
    with gpu:
        from pyiid.kernels.multi_cuda import get_normalization_array, \
            get_d_array, get_r_array, get_fq_p0_1_2, get_fq_p3, \
            fq_grad_position3, fq_grad_position5, fq_grad_position7, \
            fq_grad_position_final1, fq_grad_position_final2
        # cuda info
        stream = cuda.stream()
        stream2 = cuda.stream()
        stream3 = cuda.stream()

        # two kinds of test_kernels; NxN or NxNxQ
        # NXN
        elements_per_dim_2 = [m, n]
        tpb_l_2 = [32, 32]
        bpg_l_2 = []
        for e_dim, tpb in zip(elements_per_dim_2, tpb_l_2):
            bpg_l_2.append(int(math.ceil(float(e_dim) / tpb)))

        # NxNxQ
        elements_per_dim_3 = [m, n, qmax_bin]
        tpb_l_3 = [16, 16, 4]
        bpg_l_3 = []
        for e_dim, tpb in zip(elements_per_dim_3, tpb_l_3):
            bpg_l_3.append(int(math.ceil(float(e_dim) / tpb)))

        # START CALCULATIONS---------------------------------------------------
        dscat = cuda.to_device(scatter_array, stream2)
        dnorm = cuda.to_device(data[2], stream2)

        '--------------------------------------------------------------'
        get_normalization_array[bpg_l_3, tpb_l_3, stream2](dnorm, dscat, n_cov)
        '--------------------------------------------------------------'
        dd = cuda.to_device(data[0], stream)
        dr = cuda.to_device(data[1], stream)
        dfq = cuda.to_device(data[3], stream)
        dq = cuda.to_device(q, stream)

        get_d_array[bpg_l_2, tpb_l_2, stream](dd, dq, n_cov)
        # cuda.synchronize()

        get_r_array[bpg_l_2, tpb_l_2, stream](dr, dd)

        '--------------------------------------------------------------'
        get_fq_p0_1_2[bpg_l_3, tpb_l_3, stream](dfq, dr, qbin)
        '--------------------------------------------------------------'
        dcos_term = cuda.to_device(data[5], stream2)
        # cuda.synchronize()


        get_fq_p3[bpg_l_3, tpb_l_3, stream](dfq, dnorm)
        fq_grad_position3[bpg_l_3, tpb_l_3, stream3](dcos_term, dr, qbin)
        dgrad_p = cuda.to_device(data[4], stream2)
        # cuda.synchronize()


        # cuda.synchronize()

        fq_grad_position_final1[bpg_l_3, tpb_l_3, stream](dgrad_p, dd, dr)
        fq_grad_position5[bpg_l_3, tpb_l_3, stream2](dcos_term, dnorm)
        # cuda.synchronize()

        fq_grad_position7[bpg_l_3, tpb_l_3, stream](dcos_term, dfq, dr)
        # cuda.synchronize()

        fq_grad_position_final2[bpg_l_3, tpb_l_3, stream](dgrad_p, dcos_term)
        dgrad_p.to_host(stream)

        grad_q.append(data[4].sum(axis=1))
        index_list.append(n_cov)
        del data, dscat, dnorm, dd, dr, dfq, dcos_term, dgrad_p


def wrap_fq_grad(atoms, qmax=25., qbin=.1):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(meminfo[0])
    sort_gpus = [x for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    sort_gmem = [y for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    gpu_total_mem = sum(sort_gmem)
    total_req_mem = 6*qmax_bin*n*n + qmax_bin*n + 4*n*n + 3*n

    grad_q = []
    index_list = []
    p_dict = {}
    n_cov = 0


    # if total_req_mem < gpu_total_mem:
    if False:
        # Then the total gpu space is larger than our problem, we should give
        # out atoms by the size of the free memory, giving each gpu some work
        # to do based on its capacity.
        gpu_m = [int(round(n*float(g_mem)/gpu_total_mem)) for g_mem in sort_gmem]
        for gpu, m in zip(sort_gpus, gpu_m):
            if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                p = Thread(
                    target=sub_grad, args=(
                        gpu, q, scatter_array,
                        grad_q, qmax_bin, qbin, m, n_cov, index_list))
                p_dict[gpu] = p
                p.start()
                n_cov += m
                if n_cov == n:
                    break
        assert n_cov == n
    else:
        while n_cov < n:
            for gpu, mem in zip(sort_gpus, sort_gmem):
                m = int(math.floor(float(-4 * n * qmax_bin - 12 * n + .8 * mem) / (
                    8 * n * (3 * qmax_bin + 2))))
                if m > n - n_cov:
                    m = n - n_cov
                if gpu not in p_dict.keys() or p_dict[gpu].is_alive() is False:
                    p = Thread(
                        target=sub_grad, args=(
                            gpu, q, scatter_array,
                            grad_q, qmax_bin, qbin, m, n_cov, index_list))
                    p_dict[gpu] = p
                    p.start()
                    n_cov += m
                    if n_cov == n:
                        break
    for value in p_dict.values():
        value.join()
    # Sort grads to make certain indices are in order
    sort_grads = [x for (y, x) in sorted(zip(index_list, grad_q))]

    len(sort_grads)
    # sorted_sum_grads = [x.sum(axis=(1)) for x in sort_grads]

    # Stitch arrays together
    if len(sort_grads) > 1:
        grad_p_final = np.concatenate(sort_grads, axis=0)
    else:
        grad_p_final = sort_grads[0]


    # sum down to 1D array
    grad_p = grad_p_final

    # sum reduce to 1D
    na = np.average(scatter_array, axis=0)**2 * n

    old_settings = np.seterr(all='ignore')
    for tx in range(n):
        for tz in range(3):
            # grad_p[tx, tz, :qmin_bin] = 0.0
            grad_p[tx, tz] = np.nan_to_num(
                1 / na * grad_p[tx, tz])
    np.seterr(**old_settings)
    return grad_p

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('''
    from ase.atoms import Atoms
    import os
    from pyiid.wrappers.serial_cpu_wrap import wrap_atoms
    import matplotlib.pyplot as plt

    # n = 400
    # pos = np.random.random((n, 3)) * 10.
    # atoms = Atoms('Au' + str(n), pos)
    atoms = Atoms('Au4', [[0,0,0],[3,0,0],[0,3,0],[3,3,0]])
    wrap_atoms(atoms)

    fq = wrap_fq(atoms)
    grad_fq = wrap_fq_grad(atoms)
    print grad_fq
    plt.plot(fq), plt.show()