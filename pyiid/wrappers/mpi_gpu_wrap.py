__author__ = 'christopher'
import numpy as np
import inspect
from numba import cuda
import math
import sys
import os
sys.path.extend(['/mnt/work-data/dev/pyIID'])

from pyiid.kernels.serial_kernel import get_pdf_at_qmin, grad_pdf, get_rw, \
    get_grad_rw, get_chi_sq, get_grad_chi_sq
from mpi4py import MPI
import pyiid.wrappers.mpi.mpi_gpu_avail as mpi_gpu_avail
import pyiid.wrappers.mpi.mpi_fq_worker as mpi_fq_worker
import pyiid.wrappers.mpi.mpi_grad_worker as mpi_grad_worker


def wrap_fq(atoms, qmax, qbin):

    # get information for FQ transformation
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    #get nodes used
    n_nodes = os.getenv('NODECOUNT', 1)
    print 'nodes', n_nodes

    # get info on our gpu setup and memory requrements
    avail_loc = inspect.getfile(mpi_gpu_avail)
    comm = MPI.COMM_WORLD.Spawn(sys.executable,
        args=[avail_loc],
        maxprocs=n_nodes
    )
    # reports = comm.gather(root=MPI.ROOT)
    ranks =  comm.gather(root=MPI.ROOT)
    print ranks
    mem_list = comm.gather(root=MPI.ROOT)
    print mem_list
    comm.Disconnect()
    gpu_total_mem = 0
    for i in range(ranks[-1]+1):
        print mem_list[i]
        gpu_total_mem += mem_list[i]
    print gpu_total_mem
    total_req_mem = (2*qmax_bin*n*n+qmax_bin*n+4*n*n+3*n)*4

    # starting buffers
    fq_q = []
    n_cov = 0

    m_list = []
    while n_cov < n:
        for mem in mem_list:
            m = int(math.floor(float(-4 * n * qmax_bin - 12 * n + .8 * mem) / (
                    8 * n * (qmax_bin + 2))))
            if m > n - n_cov:
                m = n - n_cov
            m_list.append(m)
            if n_cov >= n:
                break
            n_cov += m
            if n_cov >= n:
                break
    assert sum(m_list) == n
    # The total amount of work is greater than the sum of our GPUs, no
    # special distribution needed, just keep putting problems on GPUs until
    # finished.
    kernel_loc = inspect.getfile(mpi_fq_worker)
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[kernel_loc],
        maxprocs=n_nodes
    )
    n_cov = 0
    status = MPI.Status()
    m_list += ([StopIteration] * n_nodes)
    for m in m_list:
        if m is StopIteration:
            msg = m
        else:
            msg = (q, scatter_array, qmax_bin, qbin, m, n_cov)
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=msg, dest=status.Get_source())
    print 'done master'
    reports = comm.gather(root=MPI.ROOT)
    print reports

    fq = np.zeros(qmax_bin)
    for ele in reports:
        fq[:] += ele
    na = np.average(scatter_array, axis=0)**2 * n
    old_settings = np.seterr(all='ignore')
    fq = np.nan_to_num(1 / na * fq)
    np.seterr(**old_settings)
    return fq


def wrap_pdf(atoms, qmax=25., qmin=0.0, qbin=.1, rmax=40., rstep=.01):
    """
    Generate the atomic pair distribution function

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic config   uration
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    pdf0:1darray
        The atomic pair distributuion function
    fq:1darray
        The reduced structure function
    """
    qmin_bin = int(qmin / qbin)
    fq = wrap_fq(atoms, qmax, qbin)
    fq[:qmin_bin] = 0
    pdf0 = get_pdf_at_qmin(fq, rstep, qbin, np.arange(0, rmax, rstep))
    return pdf0, fq


def wrap_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40.,
            rstep=.01):
    """
    Generate the Rw value

    Parameters
    -----------
    :param rmin:
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    rw: float
        The Rw value in percent
    scale: float
        The scale factor between the observed and calculated PDF
    pdf0:1darray
        The atomic pair distributuion function
    fq:1darray
        The reduced structure function
    """
    g_calc, fq = wrap_pdf(atoms, qmax, qmin, qbin, rmax, rstep)
    g_calc = g_calc[math.floor(rmin/rstep):]
    rw, scale = get_rw(gobs, g_calc, weight=None)
    return rw, scale, g_calc, fq


def wrap_chi_sq(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0,
                rmax=40., rstep=.01):
    """
    Generate the Rw value

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    rw: float
        The Rw value in percent
    scale: float
        The scale factor between the observed and calculated PDF
    pdf0:1darray
        The atomic pair distributuion function
    fq:1darray
        The reduced structure function
    """
    g_calc, fq = wrap_pdf(atoms, qmax, qmin, qbin, rmax, rstep)
    g_calc = g_calc[math.floor(rmin/rstep):]
    rw, scale = get_chi_sq(gobs, g_calc)
    return rw, scale, g_calc, fq


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


def wrap_fq_grad_gpu(atoms, qmax=25., qbin=.1):
    # atoms info
    q = atoms.get_positions()
    q = q.astype(np.float32)
    n = len(q)
    qmax_bin = int(qmax / qbin)
    scatter_array = atoms.get_array('scatter')

    n_nodes = os.getenv('NODECOUNT', 1)

    avail_loc = inspect.getfile(mpi_gpu_avail)
    comm = MPI.COMM_WORLD.Spawn(sys.executable,
        args=[avail_loc],
        maxprocs=n_nodes
    )
    ranks =  comm.gather(root=MPI.ROOT)
    print ranks
    mem_list = comm.gather(root=MPI.ROOT)
    print mem_list
    comm.Disconnect()
    gpu_total_mem = 0
    for i in range(ranks[-1]+1):
        print mem_list[i]
        gpu_total_mem += mem_list[i]
    print gpu_total_mem
    total_req_mem = 6*qmax_bin*n*n + qmax_bin*n + 4*n*n + 3*n

    grad_q = []
    index_list = []
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

    kernel_loc = inspect.getfile(mpi_grad_worker)
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[kernel_loc],
        maxprocs=n_nodes
    )
    n_cov = 0
    status = MPI.Status()
    m_list += ([StopIteration] * n_nodes)
    for m in m_list:
        if m is StopIteration:
            msg = m
        else:
            msg = (q, scatter_array, grad_q, qmax_bin, qbin, m, n_cov)
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=msg, dest=status.Get_source())

    reports = comm.gather(root=MPI.ROOT)

    # Sort grads to make certain indices are in order
    sort_grads = [x for (y, x) in sorted(reports)]

    len(sort_grads)

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


def wrap_grad_rw(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0, rmax=40.,
                 rstep=.01, rw=None, gcalc=None, scale=None):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    :param rmin:
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    grad_rw: float
        The gradient of the Rw value with respect to the atomic positions,
        in percent

    """
    if rw is None:
        rw, scale, gcalc, fq = wrap_rw(atoms, gobs, qmax, qmin, qbin,
                                       rmin = rmin, rmax=rmax, rstep=rstep)
    fq_grad = wrap_fq_grad_gpu(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    pdf_grad = pdf_grad[:,:,math.floor(rmin/rstep):]
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_rw(grad_rw, pdf_grad, gcalc, gobs, rw, scale, weight=None)
    return grad_rw


def wrap_grad_chi_sq(atoms, gobs, qmax=25., qmin=0.0, qbin=.1, rmin=0.0,
                     rmax=40., rstep=.01, rw=None, gcalc=None, scale=None):
    """
    Generate the Rw value gradient

    Parameters
    -----------
    atoms: ase.Atoms
        The atomic configuration
    gobs: 1darray
        The observed atomic pair distributuion function
    qmax: float
        The maximum scatter vector value
    qmin: float
        The minimum scatter vector value
    qbin: float
        The size of the scatter vector increment
    rmax: float
        Maximum r value
    rstep: float
        Size between r values

    Returns
    -------

    grad_rw: float
        The gradient of the Rw value with respect to the atomic positions,
        in percent

    """
    if rw is None:
        rw, scale, gcalc, fq = wrap_rw(atoms, gobs, qmax, qmin, qbin,
                                       rmax=rmax, rstep=rstep)
    fq_grad = wrap_fq_grad_gpu(atoms, qmax, qbin)
    qmin_bin = int(qmin / qbin)
    for tx in range(len(atoms)):
        for tz in range(3):
            fq_grad[tx, tz, :qmin_bin] = 0.
    pdf_grad = np.zeros((len(atoms), 3, rmax / rstep))
    grad_pdf(pdf_grad, fq_grad, rstep, qbin, np.arange(0, rmax, rstep))
    pdf_grad = pdf_grad[:,:,math.floor(rmin/rstep):]
    grad_rw = np.zeros((len(atoms), 3))
    get_grad_chi_sq(grad_rw, pdf_grad, gcalc, gobs, scale)
    return grad_rw


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('''
    from ase.atoms import Atoms
    import os
    from pyiid.wrappers.serial_cpu_wrap import wrap_atoms
    import matplotlib.pyplot as plt
    import sys
    sys.path.extend(['/mnt/work-data/dev/pyIID'])

    # n = 400
    # pos = np.random.random((n, 3)) * 10.
    # atoms = Atoms('Au' + str(n), pos)
    atoms = Atoms('Au4', [[0,0,0],[3,0,0],[0,3,0],[3,3,0]])
    wrap_atoms(atoms)

    # fq = wrap_fq(atoms)
    # print fq
    pdf, fq = wrap_pdf(atoms)
    grad_fq = wrap_fq_grad_gpu(atoms)
    print grad_fq
    plt.plot(pdf), plt.show()
    # for i in range(10):
    # gfq = wrap_fq_grad_gpu(atomsio)
    # ''', sort='tottime')