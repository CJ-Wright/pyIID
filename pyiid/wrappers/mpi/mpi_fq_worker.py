__author__ = 'christopher'
from mpi4py import MPI
from numba import cuda
import numpy as np
import math
import sys

if __name__ == '__main__':

    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()
    total_data = []
    for task in iter(lambda: comm.sendrecv(dest=0), StopIteration):

        q, scatter_array, qmax_bin, qbin, m, n_cov = task
        gpus = cuda.gpus.lst
        gpu = gpus[0]

        n = len(q)
        tups = [(m, n, 3), (m, n), (m, n, qmax_bin), (m, n, qmax_bin)]
        data = [np.zeros(shape=tup, dtype=np.float32) for tup in tups]
        #start GPU Processing
        with gpu:
            #import kernels
            sys.path.extend(['/mnt/work-data/dev/pyIID'])
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

            #Return Finished Data
            print 'done child'
            total_data.append(data[3].sum(axis=(0, 1)))
            del data, dscat, dnorm, dd, dq, dr, dfq
    final_data = np.asarray(total_data)
    comm.gather(sendobj=final_data.sum(axis=0), root=0)
    # Shutdown
    comm.Disconnect()
