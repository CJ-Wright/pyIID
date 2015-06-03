__author__ = 'christopher'
import numpy as np
import math
import os

from pyiid.wrappers.mpi_master import gpu_avail, mpi_fq, mpi_grad_fq
from pyiid.wrappers.nxn_atomic_gpu import atoms_per_gpu_fq, atoms_per_gpu_grad_fq

def count_nodes():
    fileloc = os.getenv("PBS_NODEFILE")
    if fileloc is None:
        return None
    else:
        with open(fileloc, 'r') as f:
            nodes = f.readlines()
        node_set = set(nodes)
        return len(node_set)


def wrap_fq(atoms, qbin=.1):

    # get information for FQ transformation
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    scatter_array = atoms.get_array('scatter')
    qmax_bin = scatter_array.shape[1]

    # get  number of allocated nodes
    n_nodes = count_nodes()
    print 'nodes', n_nodes

    # get info on our gpu setup and available memory
    ranks, mem_list = gpu_avail(n_nodes)

    # starting buffers
    n_cov = 0

    # create list of tasks
    m_list = []
    while n_cov < n:
        for mem in mem_list:
            m = atoms_per_gpu_fq(n, qmax_bin, mem)
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
    na = np.average(scatter_array, axis=0)**2 * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / na * fq)
    np.seterr(**old_settings)
    return fq


def wrap_fq_grad(atoms, qmax=25., qbin=.1):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    n_nodes = count_nodes()

    ranks, mem_list = gpu_avail(n_nodes)
    gpu_total_mem = 0
    for i in range(ranks[-1]+1):
        print mem_list[i]
        gpu_total_mem += mem_list[i]
    print gpu_total_mem
    total_req_mem = (6*qmax_bin*n*n + qmax_bin*n + 4*n*n + 3*n)*4

    n_cov = 0
    m_list = []
    while n_cov < n:
        for mem in mem_list:
            m = int(math.floor(float(-4 * n * qmax_bin - 12 * n + .8 * mem) / (
                    8 * n * (3 * qmax_bin + 2))))
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
    reports = mpi_grad_fq(n_nodes, m_list, q, scatter_array, qmax_bin, qbin)
    # list of tuples
    flat_reports = [item for sublist in reports for item in sublist]
    # seperate lists of grads and indices
    grads, indices = [x[0] for x in flat_reports], [x[1] for x in flat_reports]

    # Sort grads to make certain indices are in order
    sort_grads = [x for (y, x) in sorted(zip(indices, grads))]

    # Stitch arrays together
    if len(sort_grads) > 1:
        grad_p = np.concatenate(sort_grads, axis=0)
    else:
        grad_p = sort_grads[0]
    na = np.average(scatter_array, axis=0)**2 * n

    old_settings = np.seterr(all='ignore')
    grad_p[:, :] = np.nan_to_num(grad_p[:, :] / na)
    np.seterr(**old_settings)
    return grad_p

if __name__ == '__main__':
    # import cProfile
    # cProfile.run('''
    from ase.atoms import Atoms
    import os
    from pyiid.wrappers.scatter import wrap_atoms
    import matplotlib.pyplot as plt
    import sys
    sys.path.extend(['/mnt/work-data/dev/pyIID'])

    # n = 400
    # pos = np.random.random((n, 3)) * 10.
    # atoms = Atoms('Au' + str(n), pos)
    atoms = Atoms('Au4', [[0,0,0],[3,0,0],[0,3,0],[3,3,0]])
    wrap_atoms(atoms, exp_dict)

    # fq = wrap_fq(atoms)
    # print fq
    pdf, fq = wrap_pdf(atoms)
    grad_fq = wrap_fq_grad(atoms)
    print grad_fq
    plt.plot(pdf), plt.show()
    # for i in range(10):
    # gfq = wrap_fq_grad_gpu(atomsio)
    # ''', sort='tottime')