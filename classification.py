from spectral import *
import numpy as np
from skimage import io
import universe


spectral_library_path = "data/Hyperspectral data/Berlin Urban Gradient 2009 02 additional data/02_additional_data/spectral_library/SpecLib_Berlin_Urban_Gradient_2009.hdr"


def classify_nn(in_spectra):
    """
    Classifies the segments by nearest neighbor matching
    :param in_spectra: spectra of the segments
    :return:
    classified_labels: labels of the segments
    """

    lib = open_image(spectral_library_path)
    spectra_lib = lib.spectra
    classified_labels = []

    for i in range(0, in_spectra.shape[0]):
        px = in_spectra[i]
        px = px.reshape(1, 1, px.shape[0])
        px_dist = spectral_angles(px, spectra_lib)
        px_dist = px_dist.reshape(spectra_lib.shape[0])

        # Level1
        # names [0:30] - impervious
        # names [30:61] - veg
        # names [61:65] - soil
        # names[65:] - other
        # Level2
        # names[0:23] - roof
        # names[23:30] - pavement
        # names[30:48] - low veg
        # names[48:61] - tree
        # names[61:65] - soil
        # names[65:] - other

        px_l2_dist = []
        px_l2_dist.append(np.mean(px_dist[0:23]))
        px_l2_dist.append(np.mean(px_dist[23:30]))
        px_l2_dist.append(np.mean(px_dist[30:48]))
        px_l2_dist.append(np.mean(px_dist[48:61]))
        px_l2_dist.append(np.mean(px_dist[61:65]))
        px_l2_dist.append(np.mean(px_dist[65:75]))

        label_idx = np.argmin(px_l2_dist)
        classified_labels.append(label_idx)

    return classified_labels


def calc_precision_recall(road_seg_id_px_list, ds_name):
    """
    Calculates precision and recall by comparing the pixels labelled as road with ground truth
    :param road_seg_id_px_list: segments mapped as roads and corresponding segment id to pixel mapping
    :param ds_name: dataset name
    :return:
    precision: calculated precision value
    recall: calculated recall value
    """

    # Load GT (infrared image, roads marked with blue manually)
    gt = io.imread("data/%s_infra_GT_highroads.jpg" % ds_name)

    # Find pixels mapped as road (marked in blue) from ground truth image
    blue = np.array([0, 0, 254])
    gt_road_pxs_lst = []
    width = gt.shape[0]
    height = gt.shape[1]

    for x in range(0, width):
        for y in range(0, height):
            if np.array_equal(gt[x, y], blue):
                # gt_road_pxs_lst.append(np.array([x, y]))
                gt_road_pxs_lst.append(y * width + x)

    gt_road_pxs = np.asarray(gt_road_pxs_lst)

    # classified road pixels
    road_pxs_lst = []
    for i in range(0, len(road_seg_id_px_list)):
        road_pxs_lst += road_seg_id_px_list[i][1]
    road_pxs = np.asarray(road_pxs_lst)

    precision = 0.0
    recall = 0.0
    tp = 0
    fp = 0
    fn = 0

    if len(road_seg_id_px_list) > 0:
        tp = len(np.intersect1d(gt_road_pxs, road_pxs))
        fp = len(np.setdiff1d(road_pxs, gt_road_pxs))
        fn = len(np.setdiff1d(gt_road_pxs, road_pxs))
        precision = tp/(float(tp) + fp)
        recall = tp / (float(tp) + fn)

    print "tp, fp, fn: ", tp, fp, fn

    return precision, recall


def unmix_spectra(in_spectra):
    """
    Computes the full additive abundance vector for each segment
    :param in_spectra: spectra of the segments
    :return:
    abundance_fa: full additive abundance map for the input segment
    """
    lib = open_image(spectral_library_path)
    spectra_lib = lib.spectra
    spectra_lm = np.transpose(spectra_lib)
    inspectra_ln = np.transpose(in_spectra)
    abundance_u = np.matmul(np.matmul(np.linalg.pinv(np.matmul(spectra_lib, spectra_lm)), spectra_lib), inspectra_ln)
    z = np.ones((1, spectra_lib.shape[0]))
    part1 = np.matmul(np.linalg.pinv(np.matmul(spectra_lib, spectra_lm)), np.transpose(z))
    part11 = np.matmul(part1, np.linalg.pinv(np.matmul(z, part1)))
    part2 = np.matmul(z, abundance_u) - 1
    abundance_fa = abundance_u - np.matmul(part11, part2)
    # im = seg_mean_spectra.reshape(1,seg_mean_spectra.shape[0],seg_mean_spectra.shape[1])
    # abundance_fa = unmix(im,spectra)
    return abundance_fa


def classify_spectra_unmix(in_spectra):
    """
    Classifies the segments by linear spectral unmixing
    :param in_spectra: spectra of the segments
    :return:
    classified_labels: labels of the segments
    """
    abundance = unmix_spectra(in_spectra)
    classfied_labels = []
    for seg_id in range(0,abundance.shape[1]):
        category_id = np.argmax(abundance[:,seg_id])
        level2_category_id = map_category_level2(category_id)
        classfied_labels.append(level2_category_id)
    return classfied_labels


def map_category_level2(category_id):
    """
    map classified category_id or the band number to level 2 categories
    :param category_id: band number
    :return:
    """
    # Level2
    # names[0:23] - roof
    # names[23:30] - pavement
    # names[30:48] - low veg
    # names[48:61] - tree
    # names[61:65] - soil
    # names[65:] - other
    if 0 <= category_id < 23:
        return 0
    elif 23 <= category_id < 30:
        return 1
    elif 30 <= category_id < 48:
        return 2
    elif 48 <= category_id < 61:
        return 3
    elif 61 <= category_id < 65:
        return 4
    elif category_id >= 65:
        return 5


