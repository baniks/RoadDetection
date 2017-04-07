import numpy as np
import cv2
import math
import edge
import universe
from matplotlib import colors
import six
from skimage import measure
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def build_graph(run_flag, dist_flag, img):
    """
    buils a graph with nodes for each pixel and edges with distance weight
    :param run_flag: N for new run, E for existing run
    :param dist_flag: distance metric EU (Eucledian distance) or SAD (Spectral angle distance)
    :param img: input image
    :return:
    edges: a list of edge class object
    """

    width = img.shape[0]
    height = img.shape[1]
    num = 0
    edges = []
    weights = []

    if run_flag == 'N':

        for y in range(0, height):
            for x in range(0, width):

                if x < (width - 1):
                    a = y * width + x
                    b = y * width + (x + 1)
                    # Euclidean dist weight
                    if dist_flag == 'EU':
                        w = dist_eu(img, x, y, x + 1, y)
                        e = edge.Edge(a, b, w)
                        edges.append(e)
                        weights.append(w)
                    # SAD weight
                    if dist_flag == 'SAD':
                        w_sad = dist_sad(img, x, y, x + 1, y)
                        e_sad = edge.Edge(a, b, w_sad)
                        edges.append(e_sad)
                        weights.append(w_sad)
                    num += 1

                if y < (height - 1):
                    a = y * width + x
                    b = (y + 1) * width + x
                    # Euclidean dist weight
                    if dist_flag == 'EU':
                        w = dist_eu(img, x, y, x, y + 1)
                        e = edge.Edge(a, b, w)
                        edges.append(e)
                        weights.append(w)
                    # SAD weight
                    if dist_flag == 'SAD':
                        w_sad = dist_sad(img, x, y, x, y + 1)
                        e_sad = edge.Edge(a, b, w_sad)
                        edges.append(e_sad)
                        weights.append(w_sad)
                    num += 1

                if (x < width - 1) and (y < height - 1):
                    a = y * width + x
                    b = (y + 1) * width + (x + 1)
                    # Euclidean dist weight
                    if dist_flag == 'EU':
                        w = dist_eu(img, x, y, x + 1, y + 1)
                        e = edge.Edge(a, b, w)
                        edges.append(e)
                        weights.append(w)
                    # SAD weight
                    if dist_flag == 'SAD':
                        w_sad = dist_sad(img, x, y, x + 1, y + 1)
                        e_sad = edge.Edge(a, b, w_sad)
                        edges.append(e_sad)
                        weights.append(w_sad)
                    num += 1

                if (x < width - 1) and (y > 0):
                    a = y * width + x
                    b = (y - 1) * width + (x + 1)
                    if dist_flag == 'EU':
                        w = dist_eu(img, x, y, x + 1, y - 1)
                        e = edge.Edge(a, b, w)
                        edges.append(e)
                        weights.append(w)
                    # SAD weight
                    if dist_flag == 'SAD':
                        w_sad = dist_sad(img, x, y, x + 1, y - 1)
                        e_sad = edge.Edge(a, b, w_sad)
                        edges.append(e_sad)
                        weights.append(w_sad)
                    num += 1

        print "Graph constructed"
        print "Num of segments before merging: ", num

        # save the graph for faster run later
        f = open("output/edges_hymap02ds02_%s.dat" % dist_flag, 'w')
        f.write("%d\n" % num)
        for e in edges:
            f.write("%d %d %f\n" % (e.a, e.b, e.w))
        f.close()

    elif run_flag == 'E':

    # For existing run, load the graph from disk

        fr = open("output/edges_hymap02ds02_%s.dat" % dist_flag, 'r')
        num = int(fr.readline().split()[0])
        for line in fr:
            words = line.split()
            e = edge.Edge(int(words[0]), int(words[1]), float(words[2]))
            edges.append(e)
        fr.close()
        print "Graph loaded"
        print "Number of segments before merging: ", num

    return edges


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


def dist_eu(im, x1, y1, x2, y2):
    val = 0.0

    for d in range(0, im.shape[2]):
        val += np.round(math.pow((float(im[x1, y1, d]) - float(im[x2, y2, d])), 2),4)

    return math.sqrt(val)


def dist_sad(im,x1, y1, x2, y2):
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
def segment_graph(num_vertices, edges, c):
    print "Segmentation started..."

    # sort edges by weight
    edges = sorted(edges, key=lambda _edge: _edge.w)

    # make a disjoint-set forest
    u = universe.Universe(num_vertices)

    # init thresholds
    thres = []
    for i in range(0, num_vertices):
        thres.append(threshold(1, c))

    # for each edge, in non-decreasing weight order...
    joined = 0
    for i in range(0, len(edges)):
        pedge = edges[i]

        # components connected by this edge
        a = u.find(pedge.a)
        b = u.find(pedge.b)

        if a != b:
            if (pedge.w <= thres[a]) and (pedge.w <= thres[b]):
                # w < min(Int(C1)+threshold(C1),Int(C2+threshold(C2))
                joined += 1
                u.join(a, b)
                a = u.find(a)
                thres[a] = pedge.w + threshold(u.get_size(a), c)
                # Int(a) = max weight in a. As edge is sorted, pedge.w = max weight belonging to the comp a

    print "Components joined:", joined
    print "Segmentation complete."
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
    rgb = []
    for color in rgb_:
        rgb.append([x * 255 for x in color])

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


def get_uniq_segid_list(univ, height, width):
    """
    Returns a list of unique segment ids from the universe/segmented graph
    :param univ: universe i.e. the segmented graph
    :param height: height of the image
    :param width: width of the image
    :return:
    segid_uniq_list: list of unique segment ids
    """
    # get sorted list of segment ids
    segments = []
    for y in range(0, height):
        for x in range(0, width):
            segment = univ.find(y * width + x)
            segments.append(segment)

    segid_uniq_list = sorted(np.unique(segments))

    return segid_uniq_list


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


def get_mean_spectra(seg_px_list, img):
    """
    Calculates the mean spectra of the segments
    :param seg_px_list: list of segment to pixel mappings
    :param img: input image
    :param dim: number of channels
    :return:
    seg_mean_spectra: mean spectra of the segments
    """
    width = img.shape[0]
    dim = img.shape[2]
    seg_mean_spectra = np.ndarray(shape=(len(seg_px_list), dim), dtype=float)
    idx = 0

    for seg in seg_px_list:
        seg_px_spectra = [img[elem % width, elem / width, :] for i, elem in enumerate(seg)]
        seg_mean_spectra[idx] = np.mean(seg_px_spectra, axis=0)
        idx += 1

    return seg_mean_spectra


def map_segment_to_pxs(univ, height, width, segid_uniq_list):
    """
    Generates the mapping between the segment ids and the corresponding pixels.
    The 1st index in seg_px_list corresponds to the segment id in the 1st index of segid_uniq_list.
    :param univ: universe i.e. the segmented graph
    :param height: height of the image
    :param width: width of the image
    :param segid_uniq_list: list of unique segment ids
    :return:
    seg_px_list: list of segment to pixel mappings
    """
    seg_px_list = [[] for i in range(univ.num_sets())]

    # segment to pixel spectra grouping
    for y in range(0, height):
        for x in range(0, width):
            comp = univ.find(y * width + x)
            seg_idx = segid_uniq_list.index(comp)
            seg_px_list[seg_idx].append(y * width + x) #np.array([x, y]))

    return seg_px_list


def get_lfi(seg_px_list, segid_uniq_list, height, width):
    """
    Calculates LFI value for each segment
    :param seg_px_list: list of segment id to pixel mappings
    :param segid_uniq_list: list of unique segment ids
    :param height: height of image
    :param width: width of image
    :return:
    lfi_list: lfi values for the segments in ascending order of segment ids
    """

    idx = 0

    # create labelled image
    labelled_img = np.zeros([width, height], dtype=int)

    for seg in seg_px_list:
        # label all pixels belonging to the segment with the segment id
        for px in seg:
            x = px % width
            y = px / width
            labelled_img[x, y] = segid_uniq_list[idx]
        idx += 1

    # measure segment properties
    regions = measure.regionprops(labelled_img)

    # calculate LFI value for each segment
    lfi_list = []

    for props in regions:
        # region bounding box
        minr, minc, maxr, maxc = props.bbox

        # bounding box diagonal
        diag_sq = ((maxc-minc)*(maxc-minc)+(maxr-minr)*(maxc-minc))*1.0

        # calculate LFI as ratio of square of bbox diagonal and region area
        lfi = diag_sq / props.area
        lfi_list.append(lfi)

    print "segment LFI calculated"

    return lfi_list


def filter_shape(seg_px_list, lfi_list, thres1, thres2):
    """
    Removes segments with LFI value between threshold values thres1 and thres2
    :param seg_px_list: segment id to pixel mappings in the order of segment ids
    :param lfi_list: LFI values for the segments in the order of segment ids
    :param thres1: lower threshold value
    :param thres2: upper threshold value
    :return:
    filtered_seg_px_list: filtered segments to pixel mappings
    """
    filtered_seg_px_list = list(seg_px_list)
    cnt = 0
    for idx in range(len(lfi_list) - 1, -1, -1):
        if thres1 < lfi_list[idx] < thres2:
            cnt += 1
            # remove segment from seg_px_list
            del filtered_seg_px_list[idx]
    print "Num of regions: ", len(filtered_seg_px_list)
    return filtered_seg_px_list


def post_process(univ, edges, min_size, height, width):
    """
    performs post-processing on the segments
    :param univ: segmented graph
    :param edges: graph
    :param min_size: minimum size of segment
    :param height: height of image
    :param width: width of image
    :return:
    seg_px_list: segment id to pixel mappings in the order of segment ids
    lfi_list: LFI values for the segments in the order of segment ids
    """

    # merge_small_component()
    univ = merge_small_segments(univ, edges, min_size)

    # Get list of unique segment ids
    segid_uniq_list = get_uniq_segid_list(univ, height, width)

    # Get segment id to pixels mapping
    seg_px_list = map_segment_to_pxs(univ, height, width, segid_uniq_list)

    # Visualizing segments
    segmented_img = color_segments(seg_px_list, width, height)
    # cv2.imshow("Before Filtering", segmented_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print "Range of color: ", np.min(segmented_img), np.max(segmented_img)

    # Get LFI of segments
    lfi_list = get_lfi(seg_px_list, segid_uniq_list, height, width)

    return seg_px_list, lfi_list


def merge_small_segments(univ, edges, min_size):
    """
    merge segments that are smaller than min_size
    :param univ: segmented graph
    :param edges: graph
    :param min_size: minimum segment size
    :return:
    univ: modified universe
    """
    num = univ.num_sets()
    for i in range(0, num):
        a = univ.find(edges[i].a)
        b = univ.find(edges[i].b)
        if (a != b) and ((univ.get_size(a) < min_size) or (univ.get_size(b) < min_size)):
            univ.join(a, b)

    print "No of resultant segments: ", univ.num_sets()

    return univ


def color_segments(seg_px_list, width, height):
    # pick random colors for each component
    rgb = get_colors()
    rgb_len = len(rgb)
    output = np.empty([width, height, 3])
    idx = 0
    for seg in seg_px_list:
        for px in seg:
            x = px % width
            y = px / width
            # output[px[0] , px[1]] = rgb[idx % rgb_len]
            output[x, y] = rgb[idx % rgb_len]
        idx += 1

    return output


def post_process2(edges, candidate_seg_px_list, min_size):
    """
    :param edges: graph, list of edges at pixel level
    :param candidate_seg_px_list: candidate segments and the segment id to pixel mappings in the order of segment ids
    :param min_size: minimum segment size
    :return:

    """

    # filter road edges
    filtered_edges = filter_road_edge(edges, candidate_seg_px_list)

    # find segment edges

    # merge small neighboring segments


def filter_road_edge(edges, candidate_seg_px_list):
    """
    filter edges that
    :param edges: graph, list of edges at pixel level
    :param candidate_seg_px_list: candidate segments and the segment id to pixel mappings in the order of segment ids
    :return:
    filtered edges: edges whose vertices belong to the candidate pixel list
    """
    # for i in range(0, len(edges)):
    #   edge = edges[i]
    #

