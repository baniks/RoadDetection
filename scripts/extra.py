#!/usr/bin/python
#######################################################################
#   File name: extra.py
#   Author: Soubarna Banik
#   Description: contains few extra functions - some not in use any more
#######################################################################

from skimage.segmentation import mark_boundaries
import numpy as np
import segment_spectra as sb_spec
from skimage import measure
from skimage import io


def draw_contour(u, im, k, dist_flag):
    """
    draw contour of the segments
    :param u: universe or segmented graph
    :param im: input image
    :param k: threshold parameter
    :param dist_flag: distance metric flag
    :return:
    contoured image
    """
    rgb_im = io.imread("../images/hymap02/ds02/hymap02_ds02_sub_img_27_74_13.jpg")
    width = rgb_im.shape[0]
    height = rgb_im.shape[1]
    # r = im[:, :, 27].reshape(width, height, 1)
    # g = im[:, :, 74].reshape(width, height, 1)
    # b = im[:, :, 13].reshape(width, height, 1)
    # rgb_im = np.concatenate((r, g, b), axis=2)

    label_im = np.zeros((width, height), dtype=int)
    for y in range(0, height):
        for x in range(0, width):
            label_im[x, y] = u.find(y * width + x)
    print "label_im unique value:", len(np.unique(label_im))
    contoured_im = mark_boundaries(rgb_im, label_im, color=(0, 0, 255), mode='subpixel')

    return contoured_im


def get_mean_spectra_old(u, im, segid_uniq_list):
    """
    NOT USED ANY MORE
    :param u:
    :param im:
    :param segid_uniq_list:
    :return:
    """
    width = im.shape[0]
    height = im.shape[1]
    dim = im.shape[2]
    com_px_list = [[] for i in range(u.num_sets())]

    # segment to pixel spectra grouping
    for y in range(0, height):
        for x in range(0, width):
            comp = u.find(y * width + x)
            seg_idx = segid_uniq_list.index(comp)
            com_px_list[seg_idx].append(im[x, y, :])

    # calculating mean spectra
    seg_mean_spectra = np.ndarray(shape=(u.num_sets(), dim), dtype=float)
    idx = 0
    for seg in com_px_list:
        seg_mean_spectra[idx] = np.mean(seg, axis=0)
        idx += 1
    return seg_mean_spectra


def color_segmented_image(u, width, height, c, cat):
    """
    NOT USED ANY MORE
    :param u:
    :param width:
    :param height:
    :param c:
    :param cat:
    :return:
    """
    # pick random colors for each component
    rgb = sb_spec.get_colors()
    output = np.empty([width, height, 3])

    # get sorted list of segment ids
    comps = []
    for y in range(0, height):
        for x in range(0, width):
            comp = u.find(y * width + x)
            comps.append(comp)

    comps_u = sorted(np.unique(comps))

    for y in range(0, height):
        for x in range(0, width):
            comp = u.find(y * width + x)
            output[x, y] = rgb[comps_u.index(comp)]

    return output


def threshold_elong(size, k, u, in_comp, height, width):
    """
    calculates threshold for elongated shape
    :param size:
    :param k:
    :param u:
    :param in_comp:
    :param height:
    :param width:
    :return:
    """
    im = np.zeros([width, height], dtype=int)
    # find pixels belonging to the comp and
    # form a binary image with comp=255 and background=0
    for y in range(0, height):
        for x in range(0, width):
            comp = u.find(y * width + x)
            if comp == in_comp:
                im[x, y] = 255

    # calculate perimeter/area of region
    properties = measure.regionprops(im)
    wt = (properties[0].perimeter/properties[0].area)
    # thres=average of c/size and peri/area
    thresh = (float(k)/size + wt)/2

    return thresh


def multidim_intersect(arr1, arr2):
    """
    calculates set intersection for two multi-dimensional sets
    :param arr1: set 1
    :param arr2: set 2
    :return:
    intersection of set 1 and set 2
    """

    arr1_view = arr1.view([('', arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def multidim_difference(arr1, arr2):
    """
    calculates difference of two multi-dimensional sets
    :param arr1: set 1
    :param arr2: set 2
    :return:
    difference of set 1 and 2
    """
    arr1_view = arr1.view([('', arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)]*arr2.shape[1])
    difference = np.setdiff1d(arr1_view, arr2_view)
    return difference.view(arr1.dtype).reshape(-1, arr1.shape[1])

