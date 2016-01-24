__author__ = 'christopher'
if __name__ == '__main__':
    from mpi4py import MPI

    from pyiid.experiments.elasticscatter.gpu_wrappers.gpu_wrap import \
        atomic_fq

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    final_data = None
    for task in iter(lambda: comm.sendrecv(dest=0), StopIteration):
        if final_data is None:
            final_data = atomic_fq(*task)
        else:
            final_data += atomic_fq(*task)
    comm.gather(sendobj=final_data, root=0)

    # Shutdown process
    comm.Disconnect()
