__author__ = 'christopher'
import math


def get_d_array(d, q, n_range, range_3):
    for tx in n_range:
        for ty in n_range:
            for tz in range_3:
                d[tx, ty, tz] = q[ty, tz] - q[tx, tz]


def get_r_array(r, d, n_range):
    for tx in n_range:
        for ty in n_range:
            r[tx, ty] = math.sqrt(
                d[tx, ty, 0] ** 2 + d[tx, ty, 1] ** 2 + d[tx, ty,
                                                          2] ** 2)


def get_scatter_array(scatter_array, symbols, dpc, n_range,
                      Qmax_Qmin_bin_range, Qbin):
    for tx in n_range:
        for kq in Qmax_Qmin_bin_range:
            scatter_array[tx, kq] = dpc.scatteringfactortable.lookup(
                symbols[tx], q=kq * Qbin)


def get_fq_array(fq, r, scatter_array, n_range, Qmax_Qmin_bin_range, Qbin):
    smscale = 1
    for tx in n_range:
        for ty in n_range:
            if tx != ty:
                for kq in Qmax_Qmin_bin_range:
                    dwscale = 1
                    fq[kq] += smscale * dwscale * scatter_array[tx,
                                                            kq] * scatter_array[
                              ty, kq] / r[tx, ty] * math.sin(
                    kq * Qbin * r[tx, ty])


def get_normalization_array(norm_array, scatter_array, Qmax_Qmin_bin_range, n_range):
        for kq in Qmax_Qmin_bin_range:
            for tx in n_range:
                for ty in n_range:
                    norm_array[kq] += (scatter_array[tx, kq] * scatter_array[ty, kq])

        norm_array*1/(scatter_array.shape[0])**2

# def get_pdf_at_Qmin(qmin):