import numpy as np
import cv2
import math
import universe
from matplotlib import colors
import six
from skimage import measure
from skimage.segmentation import mark_boundaries


class Edge:
    def __init__(self, a, b, w):
        self.a = a
        self.b = b
        self.w = w


def threshold(size, c):
    return c/size


def threshold_elong(size,c,u,in_comp,height,width):
    im = np.zeros([width,height],dtype=int)
    # find pixels belonging to the comp and
    # form a binary image with comp=255 and background=0
    for y in range(0, height):
        for x in range(0, width):
            comp = u.find(y * width + x)
            if comp == in_comp:
                im[x,y] = 255

    #calculate perimeter/area of region
    properties = measure.regionprops(im)
    wt = (properties[0].perimeter/properties[0].area)
    #thres=average of c/size and peri/area
    thresh = (float(c)/size + wt)/2

    return thresh


def get_lfi(u, height, width, seg_px_list ):
    print "Calculating LFI...."
    idx = 1
    im = np.zeros([width, height], dtype=int)
    op_img = np.zeros([width, height], dtype=int)

    for comp in seg_px_list:
        # comp = list of np.array([x,y])
        # label all pixels belonging to comp with the segment number and background=0
        for px in comp:
            im[px[0], px[1]] = idx
        idx += 1

    regions = measure.regionprops(im)
    print "Num of regions: ", len(regions)

    lfi_list = []

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        diag = ((maxc-minc)*(maxc-minc)+(maxr-minr)*(maxc-minc))*1.0
        lfi = diag / props.area
        lfi_list.append(lfi)

        if lfi > 5 :
            props.label

    return lfi_list


def diff(im, x1, y1, x2, y2):
    val = 0.0

    for d in range(0, im.shape[2]):
        val += np.round(math.pow((float(im[x1, y1, d]) - float(im[x2, y2, d])), 2),4)

    return math.sqrt(val)


def diff_sad(im,x1, y1, x2, y2):
    numr = 0.0
    denom_1 = 0.0
    denom_2 = 0.0
    px_1 = im[x1, y1, :]
    px_2 = im[x2, y2, :]
    for d in range(0, im.shape[2]):
        numr += px_1[d]*px_2[d]
        denom_1 += px_1[d] * px_1[d]
        denom_2 += px_2[d] * px_2[d]
    val = math.acos(numr / (math.sqrt(denom_1) * math.sqrt(denom_2)))
    return val


# Segment image
def segment_graph(num_vertices, num_edges, edges, c):
    # sort edges by weight
    edges = sorted(edges, key=lambda edge: edge.w)

    # make a disjoint-set forest
    u = universe.Universe(num_vertices)

    # init thresholds
    thres = []
    for i in range(0,num_vertices):
        thres.append(threshold(1,c))
        # thres.append(threshold_elong(1,c,u,i,590,400 ))

    # for each edge, in non-decreasing weight order...
    joined = 0
    for i in range(0, num_edges):
        pedge = edges[i]

        # components connected by this edge
        a = u.find(pedge.a)
        b = u.find(pedge.b)
        if a != b:
            if (pedge.w <= thres[a]) and (pedge.w <= thres[b]):
                # w < min(Int(C1)+threshold(C1),Int(C2+threshold(C2))
                joined+=1
                u.join(a, b)
                a = u.find(a)
                thres[a] = pedge.w + threshold(u.get_size(a), c)
                # thres[a] = pedge.w + threshold_elong(u.get_size(a), c, u, a, 590,400)
                # Int(a) = max weight in a. As edge is sorted, pedge.w = max weight belonging to the comp a

    print "Components joined:", joined
    return u


def remove_nan_inf(im):
    cnt = 0
    for x in range(0, im.shape[0]):
        for y in range(0, im.shape[1]):
            for d in range(0, im.shape[2]):
                if not np.isfinite(im[x, y, d]):
                    cnt += 1
                    im[x, y, d] = np.mean(im[x, y, np.isfinite(im[x, y, :])])

    print "Cleaned nan/inf: ", cnt

    return im


def get_colors():
    # create color table
    colors_ = list(six.iteritems(colors.cnames))

    # Add the single letter colors.
    for name, rgb in six.iteritems(colors.ColorConverter.colors):
        hex_ = colors.rgb2hex(rgb)
        colors_.append((name, hex_))

    # Transform to hex color values.
    hex_ = [color[1] for color in colors_]
    # Get the rgb equivalent.
    rgb_ = [colors.hex2color(color) for color in hex_]
    rgb = 255*rgb_
    return rgb


def color_segmented_image(u, width, height, c, cat):
    # pick random colors for each component
    rgb = get_colors()
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


def get_mean_spectra(u, im):
    width = im.shape[0]
    height = im.shape[1]
    dim = im.shape[2]
    com_px_list = [[] for i in range(u.num_sets())]

    # get sorted list of segment ids
    comps = []
    for y in range(0, height):
        for x in range(0, width):
            comp = u.find(y * width + x)
            comps.append(comp)

    segid_list_u = sorted(np.unique(comps))

    # segment to pixel spectra grouping
    for y in range(0, height):
        for x in range(0, width):
            comp = u.find(y * width + x)
            seg_idx = segid_list_u.index(comp)
            com_px_list[seg_idx].append(im[x, y, :])

    # calculating mean spectra
    seg_mean_spectra = np.ndarray(shape=(u.num_sets(), dim), dtype=float)
    idx = 0
    for seg in com_px_list:
        seg_mean_spectra[idx] = np.mean(seg, axis=0)
        idx += 1
    return segid_list_u, seg_mean_spectra


def get_segment_to_pxs(u, height, width, segid_list_u):

    seg_px_list = [[] for i in range(u.num_sets())]

    # segment to pixel spectra grouping
    for y in range(0, height):
        for x in range(0, width):
            comp = u.find(y * width + x)
            seg_idx = segid_list_u.index(comp)
            seg_px_list[seg_idx].append(np.array([x, y]))
    # print "Length of seg_px_lst: ", len(seg_px_list)

    return seg_px_list
