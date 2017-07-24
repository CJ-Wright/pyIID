import numpy as np

__author__ = 'christopher'


def pad_pdf(pdf, rmin, rmax, rstep):
    # TODO: need functionality for PDF longer/shorter than padded pdf
    padded = np.zeros(len(pdf) + int(rmin / rstep))
    padded[int(rmin / rstep):] = pdf
    padded2 = np.zeros(len(padded) + int(np.ceil(rmax / rstep)))
    padded2[:len(padded)] = padded
    return padded2
