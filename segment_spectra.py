import numpy as np
import cv2
import misc
import edge
import universe
from matplotlib import colors
import six
from skimage import measure
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
                        w = misc.dist_eu(img, x, y, x + 1, y)
                        e = edge.Edge(a, b, w)
                        edges.append(e)
                        weights.append(w)
                    # SAD weight
                    if dist_flag == 'SAD':
                        w_sad = misc.dist_sad(img, x, y, x + 1, y)
                        e_sad = edge.Edge(a, b, w_sad)
                        edges.append(e_sad)
                        weights.append(w_sad)
                    num += 1

                if y < (height - 1):
                    a = y * width + x
                    b = (y + 1) * width + x
                    # Euclidean dist weight
                    if dist_flag == 'EU':
                        w = misc.dist_eu(img, x, y, x, y + 1)
                        e = edge.Edge(a, b, w)
                        edges.append(e)
                        weights.append(w)
                    # SAD weight
                    if dist_flag == 'SAD':
                        w_sad = misc.dist_sad(img, x, y, x, y + 1)
                        e_sad = edge.Edge(a, b, w_sad)
                        edges.append(e_sad)
                        weights.append(w_sad)
                    num += 1

                if (x < width - 1) and (y < height - 1):
                    a = y * width + x
                    b = (y + 1) * width + (x + 1)
                    # Euclidean dist weight
                    if dist_flag == 'EU':
                        w = misc.dist_eu(img, x, y, x + 1, y + 1)
                        e = edge.Edge(a, b, w)
                        edges.append(e)
                        weights.append(w)
                    # SAD weight
                    if dist_flag == 'SAD':
                        w_sad = misc.dist_sad(img, x, y, x + 1, y + 1)
                        e_sad = edge.Edge(a, b, w_sad)
                        edges.append(e_sad)
                        weights.append(w_sad)
                    num += 1

                if (x < width - 1) and (y > 0):
                    a = y * width + x
                    b = (y - 1) * width + (x + 1)
                    if dist_flag == 'EU':
                        w = misc.dist_eu(img, x, y, x + 1, y - 1)
                        e = edge.Edge(a, b, w)
                        edges.append(e)
                        weights.append(w)
                    # SAD weight
                    if dist_flag == 'SAD':
                        w_sad = misc.dist_sad(img, x, y, x + 1, y - 1)
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


def segment_graph(num_vertices, edges, c):
    """
    felzenswalb segmenation algo
    :param num_vertices: number of vertices
    :param edges: graph/ list of edges
    :param c: threshold parameter
    :return:
    univ: segmented graph
    """
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


def get_mean_spectra(seg_id_px_list, img):
    """
    Calculates the mean spectra of the segments
    :param seg_id_px_list: list of segment id to pixel mappings
    :param img: input image
    :param dim: number of channels
    :return:
    seg_mean_spectra: mean spectra of the segments
    """
    width = img.shape[0]
    dim = img.shape[2]
    seg_mean_spectra = np.ndarray(shape=(len(seg_id_px_list), dim), dtype=float)
    idx = 0

    for seg in seg_id_px_list:
        seg_px = seg[1]
        seg_px_spectra = [img[elem % width, elem / width, :] for i, elem in enumerate(seg_px)]
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
            seg_id = univ.find(y * width + x)
            seg_idx = segid_uniq_list.index(seg_id)
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

    # Merge segment id and pixel mappings
    # seg_id_px_list = zip(segid_uniq_list, seg_px_list)
    seg_id_px_arr = np.empty((univ.num_sets(), 2), dtype=object)
    for i in range(0, len(segid_uniq_list)):
        seg_id_px_arr[i][0] = segid_uniq_list[i]
        seg_id_px_arr[i][1] = seg_px_list[i]

    # Visualizing segments
    segmented_img = color_segments(seg_px_list, width, height)
    # cv2.imshow("Before Filtering", segmented_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print "Range of color: ", np.min(segmented_img), np.max(segmented_img)

    # Get LFI of segments
    lfi_list = get_lfi(seg_px_list, segid_uniq_list, height, width)

    return seg_id_px_arr, lfi_list


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


def post_process2(edges, candidate_seg_id_px_list, min_size):
    """
    :param edges: graph, list of edges at pixel level
    :param candidate_seg_id_px_list: candidate segments and the segment id to pixel mappings in the order of segment ids
    :param min_size: minimum segment size
    :return:

    """

    # filter road edges
    filtered_edges = filter_road_edge(edges, candidate_seg_id_px_list)

    # find segment edges

    # merge small neighboring segments


def filter_road_edge(edges, candidate_seg_id_px_list):
    """
    filter edges that
    :param edges: graph, list of edges at pixel level
    :param candidate_seg_id_px_list: candidate segments and the segment id to pixel mappings in the order of segment ids
    :param width: image width
    :return:
    filtered edges: edges whose vertices belong to the candidate pixel list
    """
    candidate_pxs = []
    for seg in candidate_seg_id_px_list:
        candidate_pxs += seg[1]

    filtered_edges = []
    for i in range(0, len(edges)):
        _edge = edges[i]
        if _edge.a in candidate_pxs and _edge.b in candidate_pxs:
            filtered_edges.append(_edge)

    return filtered_edges


def find_segment_edges(univ, px_edges):
    """
    finds unique inter segment edges
    :param univ: universe or segmented graph
    :param px_edges: edges whose vertices are pixels.
    :return:
    inter_seg_edges: list of inter segment edges
    """
    s_edges = []
    # convert pixel edges to segment edges
    for i in range(0, len(px_edges)):
        _edge = px_edges[i]
        seg1 = univ.find(_edge.a)
        seg2 = univ.find(_edge.b)

        if seg1 != seg2:
            s_edges.append(np.array([seg1, seg2]))
    # unique segment
    inter_seg_edges = misc.unique2d(np.asarray(s_edges))
    return inter_seg_edges


def merge_small_segments2(inter_seg_edges, seg_id_px_list, min_size):
    """

    :param inter_seg_edges: list of inter segment edges
    :param seg_id_px_list: segment to pixel mappings
    :param min_size: minimum size
    :return:
    """
    seg_size_arr = np.empty((len(seg_id_px_list), 1), dtype=int)

    for i in range(0, len(seg_id_px_list)):
        seg_size_arr[i] = len(seg_id_px_list[i][1])

    for i in range(0, len(inter_seg_edges)):
        _edge = inter_seg_edges[i]
        seg_a = _edge[0]
        seg_b = _edge[1]
        seg_a_idx = seg_id_px_list[:, 0] == seg_a
        seg_b_idx = seg_id_px_list[:, 0] == seg_b
        seg_a_sz = seg_size_arr[seg_a_idx]
        seg_b_sz = seg_size_arr[seg_b_idx]

        # if seg_b is < min_size, merge with seg_a
        if seg_a_sz >= min_size and seg_b_sz < min_size:
            # merge seg_b to seg_a in candidate_seg_id_px_list
            seg_id_px_list[seg_a_idx][0][1] += seg_id_px_list[seg_b_idx][0][1]
            # change size of seg_a in candidate_seg_id_px_list
            seg_size_arr[seg_a_idx] += seg_b_sz
            # replace seg_b in inter_seg_edges
            for i, elem in enumerate(inter_seg_edges[i+1:]):
                elem[elem == seg_b] = seg_a
            # delete seg_b from candidate_seg_id_px_list, seg_size_arr
            np.delete(seg_id_px_list,seg_b_idx,0)
            np.delete(seg_size_arr, seg_b_idx, 0)










