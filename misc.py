import numpy as np
import math


def unique2d(a):
    """
    returns unique 2d set of elements
    :param a: list of 2d elements
    :return:
    """
    x, y = a.T
    b = x + y*1.0j
    idx = np.unique(b, return_index=True)[1]
    return a[idx]


def dist_eu(im, x1, y1, x2, y2):
    """
    calculates eucledian distance between input pixels
    :param im: input image
    :param x1: pixel1 coordinate
    :param y1: pixel1 coordinate
    :param x2: pixel2 coordinate
    :param y2: pixel2 coordinate
    :return:
    eucledian distance
    """
    val = 0.0

    for d in range(0, im.shape[2]):
        val += np.round(math.pow((float(im[x1, y1, d]) - float(im[x2, y2, d])), 2),4)

    return math.sqrt(val)


def dist_sad(im,x1, y1, x2, y2):
    """
    calculates spectral angle distance between input pixels
    :param im: input image
    :param x1: pixel1 coordinate
    :param y1: pixel1 coordinate
    :param x2: pixel2 coordinate
    :param y2: pixel2 coordinate
    :return:
    spectral angle distance
    """
    numr = 0.0
    denom_1 = 0.0
    denom_2 = 0.0
    px_1 = im[x1, y1, :]
    px_2 = im[x2, y2, :]
    for d in range(0, im.shape[2]):
        numr += px_1[d]*px_2[d]
        denom_1 += px_1[d] * px_1[d]
        denom_2 += px_2[d] * px_2[d]

    cos_angle = numr / (math.sqrt(denom_1) * math.sqrt(denom_2))
    val = math.acos(min(1, max(cos_angle, -1)))
    return val


def remove_nan_inf(im):
    """
    replaces undefined channel values from input by mean of rest of the channels
    :param im: input data
    :return:
    cleaned data
    """
    cnt = 0
    for x in range(0, im.shape[0]):
        for y in range(0, im.shape[1]):
            for d in range(0, im.shape[2]):
                if not np.isfinite(im[x, y, d]):
                    cnt += 1
                    im[x, y, d] = np.mean(im[x, y, np.isfinite(im[x, y, :])])

    print "Cleaned nan/inf: ", cnt

    return im
