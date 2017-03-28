from spectral import *
import numpy as np
import cv2
import universe




def classify(in_spectra):
    lib = open_image(
        "/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 02 additional data/02_additional_data/spectral_library/SpecLib_Berlin_Urban_Gradient_2009.hdr")
    spectra_lib = lib.spectra
    # roof_mean = (np.mean(spectra[0:23,:], axis= 0)).reshape(1,111)
    # pave_mean = (np.mean(spectra[23:30,:], axis=0)).reshape(1,111)
    # veg_mean = (np.mean(spectra[30:48,:], axis=0)).reshape(1,111)
    # tree_mean = (np.mean(spectra[48:61,:], axis=0)).reshape(1,111)
    # soil_mean = (np.mean(spectra[61:65, :], axis=0)).reshape(1,111)
    # others_mean = (np.mean(spectra[65:, :], axis=0)).reshape(1,111)
    # spectra_m = np.concatenate((spectra,roof_mean,pave_mean,veg_mean,tree_mean,soil_mean,others_mean),axis=0)
    classfied_labels = []
    # print "CKPT classify in_spectra:", in_spectra.shape
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
        px_l1_dist = []
        px_l1_dist.append(np.mean(px_dist[0:23]))
        px_l1_dist.append(np.mean(px_dist[23:30]))
        px_l1_dist.append(np.mean(px_dist[30:48]))
        px_l1_dist.append(np.mean(px_dist[48:61]))
        px_l1_dist.append(np.mean(px_dist[61:65]))
        px_l1_dist.append(np.mean(px_dist[65:75]))

        # px_l1_dist.append(np.min(px_dist[0:23]))
        # px_l1_dist.append(np.min(px_dist[23:30]))
        # px_l1_dist.append(np.min(px_dist[30:48]))
        # px_l1_dist.append(np.min(px_dist[48:61]))
        # px_l1_dist.append(np.min(px_dist[61:65]))
        # px_l1_dist.append(np.min(px_dist[65:75]))

        # for i in range(75,80):
        #     px_l1_dist.append(px_dist[i])

        label_idx = np.argmin(px_l1_dist)
        classfied_labels.append(label_idx)
    return classfied_labels


def compare_gt(seg_px_list, classified_labels):
    # print "CKPT compare_gt-  seg_px_list length: ", len(seg_px_list), " classified_labels length: ", len(classified_labels)
    # print "CKPT classified labels count (pixel level):"
    pavement_pxs = [seg_px_list[i] for i, elem in enumerate(classified_labels) if elem == 1]

    pavement_pxs_lst = []
    for i in range(0,len(pavement_pxs)):
        pavement_pxs_lst+=pavement_pxs[i]
    pavement_pxs = np.asarray(pavement_pxs_lst)

    roof_pxs = [seg_px_list[i] for i, elem in enumerate(classified_labels) if elem == 0]
    roof_pxs_lst = []
    for i in range(0, len(roof_pxs)):
        roof_pxs_lst += roof_pxs[i]
    roof_pxs = np.asarray(roof_pxs_lst)
    print "Roof:", len(roof_pxs)
    print "Pavement: ", len(pavement_pxs)

    veg_pxs = [seg_px_list[i] for i, elem in enumerate(classified_labels) if elem == 2]
    veg_pxs_lst = []
    for i in range(0, len(veg_pxs)):
        veg_pxs_lst += veg_pxs[i]
    veg_pxs = np.asarray(veg_pxs_lst)
    print "veg: ",len(veg_pxs)

    tree_pxs = [seg_px_list[i] for i, elem in enumerate(classified_labels) if elem == 3]
    tree_pxs_lst = []
    for i in range(0, len(tree_pxs)):
        tree_pxs_lst += tree_pxs[i]
    tree_pxs = np.asarray(tree_pxs_lst)
    print "tree: ", len(tree_pxs)

    soil_pxs = [seg_px_list[i] for i, elem in enumerate(classified_labels) if elem == 4]
    soil_pxs_lst = []
    for i in range(0, len(soil_pxs)):
        soil_pxs_lst += soil_pxs[i]
    soil_pxs = np.asarray(soil_pxs_lst)
    print "soil: ", len(soil_pxs)

    other_pxs = [seg_px_list[i] for i, elem in enumerate(classified_labels) if elem == 5]
    other_pxs_lst = []
    for i in range(0, len(other_pxs)):
        other_pxs_lst += other_pxs[i]
    other_pxs = np.asarray(other_pxs_lst)
    print "Other: ", len(other_pxs)

    # Load GT (infrared image, roads marked with blue manually)
    gt = cv2.imread("images/hymap02_ds03_infra_E_highroads.jpg")

    blue = np.array([254, 0, 0])
    gt_road_pxs_lst = []
    for x in range(0, gt.shape[0]):
        for y in range(0, gt.shape[1]):
            if np.array_equal(gt[x, y], blue):
                gt_road_pxs_lst.append(np.array([x,y]))
    gt_road_pxs = np.asarray(gt_road_pxs_lst)

    precision = 0.0
    recall = 0.0
    tp = 0
    fp = 0
    fn = 0
    if len(pavement_pxs) > 0:
        tp = len(multidim_intersect(gt_road_pxs, pavement_pxs))
        fp = len(multidim_difference(pavement_pxs, gt_road_pxs))
        fn = len(multidim_difference(gt_road_pxs, pavement_pxs))
        precision = tp/(float(tp) + fp)
        recall = tp / (float(tp) + fn)

    print "tp, fp, fn: ", tp, fp, fn

    return precision, recall, pavement_pxs


def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def multidim_difference(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    difference = np.setdiff1d(arr1_view, arr2_view)
    return difference.view(arr1.dtype).reshape(-1, arr1.shape[1])


def unmix_spectra(in_spectra):
    lib = open_image(
        "/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 02 additional data/02_additional_data/spectral_library/SpecLib_Berlin_Urban_Gradient_2009.hdr")
    spectra_lib = lib.spectra
    spectra_lm = np.transpose(spectra_lib)
    inspectra_ln = np.transpose(in_spectra)
    abundance_u = np.matmul(np.matmul(np.linalg.pinv(np.matmul(spectra_lib,spectra_lm)),spectra_lib),inspectra_ln)
    z = np.ones((1,75))
    part1 = np.matmul(np.linalg.pinv(np.matmul(spectra_lib,spectra_lm)),np.transpose(z))
    part11 = np.matmul(part1,np.linalg.pinv(np.matmul(z,part1)))
    part2 = np.matmul(z,abundance_u) - 1
    abundance_fa = abundance_u - np.matmul(part11,part2)
    # im = seg_mean_spectra.reshape(1,seg_mean_spectra.shape[0],seg_mean_spectra.shape[1])
    # abundance_fa = unmix(im,spectra)
    return abundance_fa


def classify_spectra_unmix(abundance):
    classfied_labels = []
    for seg_id in range(0,abundance.shape[1]):
        category_id = np.argmax(abundance[:,seg_id])
        level2_category_id = map_category_level2(category_id)
        classfied_labels.append(level2_category_id)
    return classfied_labels


def map_category_level2(category_id):
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

