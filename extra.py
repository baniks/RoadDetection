from skimage.segmentation import mark_boundaries
import cv2
import numpy as np
import segment_spectra as sb_spec
from skimage import measure


def draw_contour(u,im,c,cat):
    width = im.shape[0]
    height = im.shape[1]
    r = im[:, :, 18].reshape(width, height, 1)
    g = im[:, :, 5].reshape(width, height, 1)
    b = im[:, :, 1].reshape(width, height, 1)
    rgb_im = np.concatenate((r, g, b), axis=2)

    label_im = np.zeros((width, height), dtype=int)
    for y in range(0, height):
        for x in range(0, width):
            label_im[x, y] = u.find(y * width + x)
    print "label_im unique value:", len(np.unique(label_im))
    contoured_im = mark_boundaries(rgb_im, label_im, color=(1, 1, 1))
    title = "%s segmented image %s" % (cat, c)
    cv2.imshow(title, contoured_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return contoured_im


def get_mean_spectra_old(u, im, segid_uniq_list):
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

    title = "%s segmented image %s" % (cat, c)
    # cv2.imshow(title, output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return output


def threshold_elong(size, c, u, in_comp, height, width):
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
    thresh = (float(c)/size + wt)/2

    return thresh