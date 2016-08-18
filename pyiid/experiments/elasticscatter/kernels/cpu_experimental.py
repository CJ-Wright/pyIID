from numba import jit, i4
from builtins import range
from pyiid.experiments.elasticscatter.kernels import k_to_ij

__author__ = 'christopher'


@jit(target='cpu', nopython=True)
def experimental_sum_grad_cpu(new_grad, grad, k_cov):
    for k in range(grad.shape[0]):
        i, j = k_to_ij(i4(k + k_cov))
        for qx in range(grad.shape[2]):
            for tz in range(3):
                new_grad[i, tz, qx] -= grad[k, tz, qx]
                new_grad[j, tz, qx] += grad[k, tz, qx]
