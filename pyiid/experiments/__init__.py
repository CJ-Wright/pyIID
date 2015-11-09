import math
from numba import cuda
import numpy as np

__author__ = 'christopher'


def get_gpus_mem():
    gpus = cuda.gpus.lst
    mem_list = []
    for gpu in gpus:
        with gpu:
            meminfo = cuda.current_context().get_memory_info()
        mem_list.append(int(meminfo[0]))
    sort_gpus = [x for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    sort_gmem = [y for (y, x) in sorted(zip(mem_list, gpus), reverse=True)]
    return [sort_gpus[0]], [sort_gmem[0]]
    # return sort_gpus, sort_gmem
    # return [sort_gpus[1]], [sort_gmem[1]]


def generate_grid(elements, tpb):
    assert len(elements) == len(tpb)
    bpgs = []
    for e_dim, thds in zip(elements, tpb):
        bpg = int(math.ceil(float(e_dim) / thds))
        if bpg < 1:
            bpg = 1
        bpgs.append(bpg)
    for e, t, b in zip(elements, tpb, bpgs):
        # print e, t*b
        assert (e <= t * b)
    return bpgs
