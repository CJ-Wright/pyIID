__author__ = 'christopher'
if __name__ == '__main__':
    import socket
    from mpi4py import MPI
    from numba import cuda

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    # maybe use with gpu statement

    # print 'start', rank, socket.gethostname()
    meminfo = int(cuda.current_context().get_memory_info()[0])
    # print 'finish', rank, socket.gethostname()

    cuda.close()

    comm.gather(sendobj=meminfo, root=0)
    comm.Disconnect()