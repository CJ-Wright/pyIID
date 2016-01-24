__author__ = 'christopher'
if __name__ == '__main__':
    from mpi4py import MPI
    from numba import cuda

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    meminfo = int(cuda.current_context().get_memory_info()[0])
    cuda.close()

    comm.gather(sendobj=meminfo, root=0)
    comm.Disconnect()
