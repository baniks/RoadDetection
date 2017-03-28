from spectral import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import segment_spectra as sb_spec
import spectral_library as sb_lib
import cPickle
import sys

# Parameter settings
run_flag = 'E'  # for existing run : E, for new run: N
dist_flag = 'SAD'  # SAD: SAD distance metric, EU: euclidean distance metric
sigma = 0.25
c = int(sys.argv[1])
min_size = int(sys.argv[2])

# -------------------------------- Load data -------------------------------- #

# Load whole image
# img = open_image("/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/HyMap01_Berlin_Urban_Gradient_2009.hdr")

# Load image subset
img = cPickle.load(open("data/hymap02_ds02.pkl", "rb"))

print img.shape
print "Data loaded"

# -------------------------------- Pre - processing -------------------------------- #

# Data cleaning
img = sb_spec.remove_nan_inf(img)
print "Data cleaning: removed nan/inf.."

# Smooth channels
sm_img = cv2.GaussianBlur(img, (3, 3), sigma)
print "Channels smoothed"

# final check data for nan/inf
print "Final data check:",
if False in np.isfinite(sm_img):
    print "Data could not be cleaned. Exiting..."
    exit(1)
print "Success"

# -------------------------------- Superpixel generation -------------------------------- #

width = sm_img.shape[0]
height = sm_img.shape[1]
dim = sm_img.shape[2]
num = 0
edges = []
weights = []

# ---- Graph construction ---- #
# For new run, build the graph
if run_flag == 'N':

    for y in range(0, height):
        for x in range(0, width):

            if x < (width - 1):
                a = y * width + x
                b = y * width + (x + 1)
                # Euclidean dist weight
                if dist_flag == 'EU':
                    w = sb_spec.diff(sm_img, x, y, x + 1, y)
                    e = sb_spec.Edge(a, b, w)
                    edges.append(e)
                    weights.append(w)
                # SAD weight
                if dist_flag == 'SAD':
                    w_sad = sb_spec.diff_sad(sm_img, x, y, x + 1, y)
                    e_sad = sb_spec.Edge(a, b, w_sad)
                    edges.append(e_sad)
                    weights.append(w_sad)
                num += 1

            if y < (height - 1):
                a = y * width + x
                b = (y + 1) * width + x
                # Euclidean dist weight
                if dist_flag == 'EU':
                    w = sb_spec.diff(sm_img, x, y, x, y + 1)
                    e = sb_spec.Edge(a, b, w)
                    edges.append(e)
                    weights.append(w)
                # SAD weight
                if dist_flag == 'SAD':
                    w_sad = sb_spec.diff_sad(sm_img, x, y, x, y + 1)
                    e_sad = sb_spec.Edge(a, b, w_sad)
                    edges.append(e_sad)
                    weights.append(w_sad)
                num += 1

            if (x < width - 1) and (y < height - 1):
                a = y * width + x
                b = (y + 1) * width + (x + 1)
                # Euclidean dist weight
                if dist_flag == 'EU':
                    w = sb_spec.diff(sm_img, x, y, x + 1, y + 1)
                    e = sb_spec.Edge(a, b, w)
                    edges.append(e)
                    weights.append(w)
                # SAD weight
                if dist_flag == 'SAD':
                    w_sad = sb_spec.diff_sad(sm_img, x, y, x + 1, y + 1)
                    e_sad = sb_spec.Edge(a, b, w_sad)
                    edges.append(e_sad)
                    weights.append(w_sad)
                num += 1

            if (x < width - 1) and (y > 0):
                a = y * width + x
                b = (y - 1) * width + (x + 1)
                if dist_flag == 'EU':
                    w = sb_spec.diff(sm_img, x, y, x + 1, y - 1)
                    e = sb_spec.Edge(a, b, w)
                    edges.append(e)
                    weights.append(w)
                # SAD weight
                if dist_flag == 'SAD':
                    w_sad = sb_spec.diff_sad(sm_img, x, y, x + 1, y - 1)
                    e_sad = sb_spec.Edge(a, b, w_sad)
                    edges.append(e_sad)
                    weights.append(w_sad)
                num += 1

    print "Graph constructed"
    print "Num of segments before merging: ", num, width * height * 4
    print "Min weight:", np.min(weights), "Max weight:", np.max(weights)

    # save the graph for faster run later
    f = open("output/edges_hymap02ds02_%s.dat" % dist_flag, 'w')
    f.write("%d\n" % num)
    for e in edges:
        f.write("%d %d %f\n" % (e.a, e.b, e.w))
    f.close()

# For existing run, load the graph from disk
if run_flag == 'E':
    fr = open("output/edges_hymap02ds02_%s.dat" % dist_flag, 'r')
    num = int(fr.readline().split()[0])
    for line in fr:
        words = line.split()
        e = sb_spec.Edge(int(words[0]), int(words[1]), float(words[2]))
        edges.append(e)
    fr.close()
    print "Graph loaded"

# ----Build graph ended---- #

precision_lst = []
recall_lst = []

# ----Segment graph---- #
print "Segmentation for c: ", c
print "Min_size: ", min_size
print "Distance metric: ", dist_flag

u = sb_spec.segment_graph(width*height, num, edges, c)
print "Graph segmented"
# ----Segment graph ended---- #

# ----Post-processing graph---- #

# Merge small components
for i in range(0, num):
    a = u.find(edges[i].a)
    b = u.find(edges[i].b)
    if (a != b) and ((u.get_size(a) < min_size) or (u.get_size(b) < min_size)):
        u.join(a, b)

print "%s segments merged" % dist_flag
print "No of resultant segments: ", u.num_sets()

# --------------------------------Shape filtering---------------------------------------- #

# Get list of unique segment ids and mean spectra of the segments
segid_list_u, seg_mean_spectra = sb_spec.get_mean_spectra(u, sm_img)
seg_px_list = sb_spec.get_segment_to_pxs(u, height, width, segid_list_u)

# Calculate lfi of the segments
lfi_list = sb_spec.get_lfi(u, height, width, seg_px_list)
print "lfi min: ", min(lfi_list)
print "lfi max: ", max(lfi_list)
# the histogram of the lfi
n, bins, patches = plt.hist(lfi_list, normed=0, facecolor='green', alpha=0.75)
plt.grid(True)
plt.show()

# TEST: eliminate segments with LFI<5


# --------------------------------Superpixel generation ends-------------------------------- #

# --------------------------------Visualization-------------------------------- #

# Color each segment
colored_im = sb_spec.color_segmented_image(u, width, height, c, dist_flag)
save_rgb("output/shell output/threshold/Hymap02_ds02_%s_segmented%s.jpg" % (dist_flag, c), colored_im)

# contoured_im = sb_spec.draw_contour(u, img, c, dist_flag)
# save_rgb("output/shell output/Hymap02_ds02_test_contour_%s_segmented%s.jpg" % (dist_flag, c), contoured_im, bands=[0, 1, 2])

# save input image in rgb
save_rgb("output/shell output/threshold/hymap02ds02_input_bigfile.jpg", img, bands=[27, 74, 13])



# classify by distance from spectral library
classified_labels = sb_lib.classify(seg_mean_spectra)

# print "CKPT classified labels count for superpixels:"
# print "Roof: ", classified_labels.count(0)
# print "Pavement: ", classified_labels.count(1)
# print "Veg: ", classified_labels.count(2)
# print "Tree: ", classified_labels.count(3)
# print "Soil: ", classified_labels.count(4)
# print "Other: ", classified_labels.count(5)

# Spectral unmixing
# abundance_fa = sb_lib.unmix_spectra(seg_mean_spectra)
# print abundance_fa.shape
# cPickle.dump(abundance_fa, open("output/abundance_fa.pkl", "wb"))

# classify by abundance fractions
# classified_labels = sb_lib.classify_spectra_unmix(abundance_fa)

# compare with ground truth
precision, recall, pavement_pxs = sb_lib.compare_gt(seg_px_list, classified_labels)

print "precision: ", precision
print "recall: ", recall

op_img = sm_img[:, :, 39:42]
red = np.array([0, 0, 254])
blue = np.array([255, 0, 0])
print pavement_pxs.shape
for p in range(0,pavement_pxs.shape[0]):
    op_img[pavement_pxs[p][0], pavement_pxs[p][1]] = red

# view = imshow(op_img,bands=(0,1,2,))
# cv2.imshow("classified image", op_img)
cv2.imwrite("output/shell output/threshold/op_hymap02ds02_SAD_mean_%s_%s.jpg" % (c, min_size), op_img)
# gt = cv2.imread("images/hymap02_ds03_infra_E.jpg")
# cv2.imshow("gt",gt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


print "End"
# print "Summary:"
# print "Distance measure: ", dist_flag
# print "Spectral library match - distance grouped by level2"
# print "min_size: ", min_size_list
# print "c: ", c_list
# print "precision: ", precision_lst
# print "recall: ", recall_lst
################################################################################################


        # # clustering EU segments
        # centers, labels = vq.kmeans2(seg_mean_spectra, 4, 40)
        # # use vq() to get as assignment for each observation/segment mean spectra.
        # assignment, cdist = vq.vq(seg_mean_spectra, centers)
        #
        # plt.scatter(seg_mean_spectra[:, 0], seg_mean_spectra[:, 9], c=assignment)
        # plt.ylabel('Band 9 Reflectance (%)')
        # plt.xlabel("Band 0 Reflectance (%)")
        # plt.title("Mean reflectance of bands 0 and 9 for all segments")
        # plt.show()
        #
        # plt.figure()
        # plt.hold(1)
        # for i in range(centers.shape[0]):
        #     plt.plot(centers[i])
        #
        # plt.title('Spectra of cluster centers')
        # plt.ylabel('Reflectance (%)')
        # plt.xlabel("Wavelength (Micrometers)")
        # plt.show()
        #
        # # spectral_angles(sub_img[100:101, 200:201, :], sub_img[150:151, 200])
        #
        # # classifiy culster centers
        # classified_labels = sb_lib.classify(centers)

        # compare with GT



