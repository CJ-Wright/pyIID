import sys
import inspect
from threading import Thread

from numba import cuda

import pyiid.experiments.elasticscatter.mpi_wrappers.mpi_gpu_avail as \
    mpi_gpu_avail
import pyiid.experiments.elasticscatter.mpi_wrappers.mpi_fq_worker as \
    mpi_fq_worker
import pyiid.experiments.elasticscatter.mpi_wrappers.mpi_grad_worker as \
    mpi_grad_worker
from pyiid.experiments.elasticscatter.gpu_wrappers.gpu_wrap import subs_fq

__author__ = 'christopher'


def gpu_avail(n_nodes):
    """
    For each node in the allocated nodes get the memory for each GPU

    Parameters
    ----------
    n_nodes: int
        Number of allocated nodes, not including the head node
    Returns
    -------
    list of floats:
        Amount of memory per GPU
    """
    from mpi4py import MPI

    avail_loc = inspect.getfile(mpi_gpu_avail)
    comm = MPI.COMM_WORLD.Spawn(sys.executable,
                                args=[avail_loc],
                                maxprocs=n_nodes
                                )
    mem_list = comm.gather(root=MPI.ROOT)
    comm.Disconnect()
    return mem_list


def mpi_fq(n_nodes, m_list, q, scatter_array, qbin):
    """
    Breakup the job across the GPU enabled nodes

    Parameters
    ----------
    n_nodes: int
        Number of allocated nodes, not including the head node
    Returns
    -------
    list of floats:
        Amount of memory per GPU
    """
    from mpi4py import MPI

    kernel_loc = inspect.getfile(mpi_fq_worker)
    comm = MPI.COMM_WORLD.Spawn(
        sys.executable,
        args=[kernel_loc],
        maxprocs=n_nodes
    )
    n_cov = 0
    status = MPI.Status()
    m_list += ([StopIteration] * n_nodes)
    p = None
    thread_q = []
    for m in m_list:
        if m is StopIteration:
            msg = m
        else:
            msg = (q, scatter_array, qbin, m, n_cov)

        # If the thread on the main node is done, or not started:
        # give a problem to it
        if p is None or p.is_alive() is False:
            cuda.close()
            p = Thread(
                target=subs_fq, args=(
                    cuda.gpus.lst[0], q, scatter_array, thread_q,
                    qbin, m,
                    n_cov))
            p.start()
        else:
            comm.recv(source=MPI.ANY_SOURCE, status=status)
            comm.send(obj=msg, dest=status.Get_source())
        if type(m) == int:
            n_cov += m
    p.join()
    # Make certain we have covered all the atoms
    assert n_cov == len(q)
    # TODO: Make Numpy based Gather for faster memory transfer or Sum Reduce
    reports = comm.gather(root=MPI.ROOT)
    comm.Disconnect()
    reports += thread_q
    return reports


def mpi_grad_fq(n_nodes, m_list, q, scatter_array, qbin):
    """
    Breakup the grad F(Q) job across GPU enabled nodes

    Parameters
    ----------
    n_nodes: int
        Number of allocated nodes, not including the head node
    Returns
    -------
    list of floats:
        Amount of memory per GPU
    """
    from mpi4py import MPI

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
            msg = (q, scatter_array, qbin, m, n_cov)
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=msg, dest=status.Get_source())
    # TODO: Make Numpy based Gather for faster memory transfer
    reports = comm.gather(root=MPI.ROOT)
    return reports
