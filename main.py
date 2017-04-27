from spectral import *
import numpy as np
import cv2
import cPickle
import segment_spectra as sb_spec
import classification as sb_lib
import misc
import matplotlib.pyplot as plt

# Parameter settings
run_flag = 'E'  # for existing run : E, for new run: N
dist_flag = 'SAD'  # SAD: SAD distance metric, EU: euclidean distance metric
sigma = 0.25
c = int(sys.argv[1])
min_size = int(sys.argv[2])
ds_name = str(sys.argv[3])

print "Parameters:"
print "c: ", c
print "Min_size: ", min_size
print "Distance metric: ", dist_flag

# -------------------------------- Load data -------------------------------- #

# Load whole image
# img = open_image("/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/HyMap02_Berlin_Urban_Gradient_2009.hdr")

# Load image subset
img = cPickle.load(open("data/%s.pkl" % ds_name, "rb"))

print "Image dimension:", img.shape
print "Data loaded"

# -------------------------------- Pre - processing -------------------------------- #

# Data cleaning
# img = misc.remove_nan_inf(img)
# print "Data cleaning: removed nan/inf.."

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

# STEP1: Graph construction
edges = sb_spec.build_graph(run_flag, dist_flag, sm_img, ds_name)
print "Checkpoint : build graph ended."

# STEP2: Segment graph/Generate superpixels
univ = sb_spec.segment_graph(width*height, edges, c)
print "Checkpoint : segment graph ended."

# STEP3: Post-processing segments/superpixels
# detect small components as the roads are very thin. K and min_size controls the overall size
# and hence large k and min_size won't detect the thin line segments.
seg_id_px_arr = sb_spec.post_process(univ, edges, min_size, height, width)
print "Checkpoint : post-process 1 ended."

segmented_img = sb_spec.color_segments(seg_id_px_arr[:, 1], width, height)
cv2.imwrite("output/shell output/perim_area/ds02/segmented_image_c_%s_sz_%s.jpg" % (c, min_size), segmented_img)

# STEP4: Calculate mean spectra of the segments/superpixels
seg_mean_spectra = sb_spec.get_mean_spectra(seg_id_px_arr, sm_img)
print "Checkpoint : mean spectra ended."

# --------------------------------Superpixel generation ends-------------------------------- #

# --------------------------------Classification-------------------------------- #
# STEP5: Classify by nearest neighbor from the spectral library
classified_labels = sb_lib.classify(seg_mean_spectra)
print "Checkpoint : classify by NN ended."

# Filter label roads (asphalt/concrete)
candidate_seg_id_px_arr = [seg_id_px_arr[i] for i, elem in enumerate(classified_labels) if elem == 1]
candidate_seg_id_px_arr = np.asarray(candidate_seg_id_px_arr)
print "# of segments labelled as road: ", len(candidate_seg_id_px_arr)
print "Checkpoint : filter by road label ended."


# STEP 6: post process 2
# small k and min_size forces to detect small segments -> with low LFI.
# merge the connected segments that are small. then calculate lfi and filter shape
merged_candidate_seg_id_px_arr = sb_spec.post_process2(univ, edges, candidate_seg_id_px_arr, 30)
merged_candidate_seg_id_px_arr = merged_candidate_seg_id_px_arr[np.argsort(merged_candidate_seg_id_px_arr[:, 0])]
print "Checkpoint : post-process 2 ended."

# merged_candidate_seg_id_px_arr = cPickle.load(open("data/merged_candidate_seg_id_px_arr.pkl", "rb"))

# color_merged_segs = sb_spec.color_segments(merged_candidate_seg_id_px_arr[:, 1], width, height)
# cv2.imwrite("output/shell output/LFI/post_process2_op_0.png", color_merged_segs)

# STEP 7: calculate LFI of segments
# lfi_list = sb_spec.get_lfi(merged_candidate_seg_id_px_arr, height, width)
shp_score_list, label_list = sb_spec.get_perim_area(merged_candidate_seg_id_px_arr, height, width)
print "Checkpoint : calculate LFI ended."

print "Perimeter/Area score min: ", min(shp_score_list), "max: ", max(shp_score_list)

hist, bins = np.histogram(shp_score_list, 10)
w = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=w)
plt.savefig("output/shell output/perim_area/ds02/shape_score_hist_%s_%s" % (c, min_size))

# Test LFI
shp_score_img = np.zeros([width, height, 3], int)
col = np.empty([3], int)
for idx in range(len(shp_score_list) - 1, -1, -1):
    px_list = merged_candidate_seg_id_px_arr[idx][1]
    if 0 <= shp_score_list[idx] < 0.10:
        col = np.array([143, 143, 188]) # rose
    elif 0.10 <= shp_score_list[idx] < 0.20:
        col = np.array([255, 0, 255]) # magenta
    elif 0.20 <= shp_score_list[idx] < 0.30:
        col = np.array([0, 69, 255])  # orange
    elif 0.30 <= shp_score_list[idx] < 0.40:
        col = np.array([255, 255, 0])   # cyan
    elif 0.40 <= shp_score_list[idx] < 0.50:
        col = np.array([0, 255, 0]) # lime
    elif 0.50 <= shp_score_list[idx] < 0.60:
        col = np.array([0, 0, 255])   # red
    elif 0.6 <= shp_score_list[idx] < 0.70:
        col = np.array([255, 0, 0])  # blue
    elif 0.7 <= shp_score_list[idx] < 0.8:
        col = np.array([0, 128, 0])   # green
    elif 0.8 <= shp_score_list[idx] < 0.9:
        col = np.array([128, 0, 128])   # purple
    elif 0.9 <= shp_score_list[idx] < 1.0:
        col = np.array([0, 255, 255])  # chocolate
    elif shp_score_list[idx] >= 1.0:
        col = np.array([255, 255, 255])  # white

    for px in px_list:
        x = px % width
        y = px / width
        shp_score_img[x, y] = col

cv2.imwrite("output/shell output/perim_area/ds02/score_colors_%s_%s_PP_30.jpg" % (c, min_size), shp_score_img)

# STEP8: Final filtering based on shape
road_seg_id_px_arr = sb_spec.filter_shape(merged_candidate_seg_id_px_arr, shp_score_list, label_list, 0, 0.6)
print "# of road segments after shape filtering: ", len(road_seg_id_px_arr)
print "Checkpoint : filter by shape ended."

# STEP7: Compare with ground truth
precision_lst = []
recall_lst = []
precision, recall, road_pxs = sb_lib.calc_precision_recall(road_seg_id_px_arr, ds_name)

print "Precision: ", precision
print "Recall: ", recall

# STEP8: Save classified road as output image
op_img = np.zeros((width, height, 3), int)
red = np.array([0, 0, 255])

for px in road_pxs:
    op_img[px % width, px / width] = red

cv2.imwrite("output/shell output/perim_area/ds02/road_op_%s_SAD_%s_%s_PP2_30.jpg" % (ds_name, c, min_size), op_img)

# saving output
recall_file = open("output/shell output/perim_area/ds02/recall_%s.txt" % min_size, 'a')
recall_file.write(str(recall))
recall_file.write("\n")
prec_file = open("output/shell output/perim_area/ds02/prec_%s.txt" % min_size, 'a')
prec_file.write(str(precision))
prec_file.write("\n")
recall_file.close()
prec_file.close()
