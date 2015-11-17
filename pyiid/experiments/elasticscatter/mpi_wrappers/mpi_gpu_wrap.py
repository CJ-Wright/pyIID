import os

import numpy as np
from numba import cuda

from pyiid.experiments.elasticscatter.mpi_wrappers.mpi_master import \
    gpu_avail, mpi_fq, mpi_grad_fq
from pyiid.experiments.elasticscatter.gpu_wrappers.gpu_wrap import \
    gpu_fq_atoms_allocation, atoms_per_gpu_grad_fq
__author__ = 'christopher'


def count_nodes():
    fileloc = os.environ["PBS_NODEFILE"]
    if fileloc is None:
        return None
    else:
        with open(fileloc, 'r') as f:
            nodes = f.readlines()
        node_set = set(nodes)
        # give back the number of nodes, not counting the head node
        return len(node_set) - 1


def wrap_fq(atoms, qbin=.1, sum_type='fq'):
    # get information for FQ transformation
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    qmax_bin = scatter_array.shape[1]

    # get  number of allocated nodes
    n_nodes = count_nodes()
    print 'nodes', n_nodes

    # get info on our gpu setup and available memory
    mem_list = gpu_avail(n_nodes)
    mem_list.append(cuda.current_context().get_memory_info()[0])

    # starting buffers
    n_cov = 0

    # create list of tasks
    m_list = []
    while n_cov < n:
        for mem in mem_list:
            m = gpu_fq_atoms_allocation(mem, n, qmax_bin)
            if m > n - n_cov:
                m = n - n_cov
            m_list.append(m)
            if n_cov >= n:
                break
            n_cov += m
            if n_cov >= n:
                break

    # Make certain that we have covered all the atoms
    assert sum(m_list) == n

    reports = mpi_fq(n_nodes, m_list, q, scatter_array, qbin)

    fq = np.zeros(qmax_bin)
    for ele in reports:
        fq[:] += ele
    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / na * fq)
    np.seterr(**old_settings)
    return fq


def wrap_fq_grad(atoms, qbin=.1, sum_type='fq'):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    if sum_type == 'fq':
        scatter_array = atoms.get_array('F(Q) scatter')
    else:
        scatter_array = atoms.get_array('PDF scatter')
    qmax_bin = scatter_array.shape[1]

    n_nodes = count_nodes()

    ranks, mem_list = gpu_avail(n_nodes)
    gpu_total_mem = 0
    for i in range(ranks[-1] + 1):
        print mem_list[i]
        gpu_total_mem += mem_list[i]
    print gpu_total_mem

    n_cov = 0
    m_list = []
    while n_cov < n:
        for mem in mem_list:
            m = atoms_per_gpu_grad_fq(mem, n, qmax_bin)
            if m > n - n_cov:
                m = n - n_cov
            m_list.append(m)
            if n_cov >= n:
                break
            n_cov += m
            if n_cov >= n:
                break
    assert sum(m_list) == n

    # list of list of tuples, I think
    reports = mpi_grad_fq(n_nodes, m_list, q, scatter_array, qbin)
    grad_p = np.zeros((n, 3, qmax_bin), dtype=np.float32)
    for r in reports:
        grad_p += r

    na = np.average(scatter_array, axis=0) ** 2 * n
    old_settings = np.seterr(all='ignore')
    grad_p[:, :] = np.nan_to_num(grad_p[:, :] / na)
    np.seterr(**old_settings)
    return grad_p
