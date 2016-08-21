__author__ = 'christopher'
if __name__ == '__main__':
    from mpi4py import MPI

    from pyiid.experiments.elasticscatter.gpu_wrappers.gpu_wrap import \
        atomic_grad_fq

    grad_cov = None
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    for task in iter(lambda: comm.sendrecv(dest=0), StopIteration):
        if grad_cov is None:
            grad_cov = atomic_grad_fq(*task, adps)
        else:
            grad_cov += atomic_grad_fq(*task, adps)
    # Return Finished Data
    # TODO: Use MPI Buffer object to do this for speed enhancement
    comm.gather(sendobj=grad_cov, root=0)
    # Shutdown
    comm.Disconnect()
