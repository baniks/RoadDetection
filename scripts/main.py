#!/usr/bin/python
#######################################################################
#   File name: main.py
#   Author: Soubarna Banik
#   Description: main script for super-pixel based road detection
#######################################################################

from spectral import *
import numpy as np
import cPickle
import segment_spectra as sb_spec
import classification as sb_lib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Parameter settings
run_flag = 'E'  # for existing run : E, for new run: N
dist_flag = 'SAD'  # SAD: SAD distance metric, EU: euclidean distance metric
sigma = 0.25
# k = int(sys.argv[1])
# min_size = int(sys.argv[2])
# ds_name = str(sys.argv[3])

k = 100
min_size = 10
min_size2 = 30
ds_name = "hymap02_ds02_sub_img1"
data_dir = "../data"
output_dir = "../output/ds02"

print "Parameters:"
print "k: ", k
print "Min_size: ", min_size
print "Distance metric: ", dist_flag
print "_________________________________________________________"
# -------------------------------- Load data -------------------------------- #

# Load whole image
# img = open_image("%s/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/HyMap02_Berlin_Urban_Gradient_2009.hdr" % data_dir)

# Load image subset
img = cPickle.load(open("%s/%s.pkl" % (data_dir, ds_name), "rb"))

print "Image dimension:", img.shape
print "Data loaded"

# -------------------------------- Pre - processing -------------------------------- #

# Data cleaning
# img = misc.remove_nan_inf(img)
# print "Data cleaning: removed nan/inf.."

# Smooth channels
sm_img = gaussian_filter(img, sigma)
print "Channels smoothed"

# final check data for nan/inf
print "Data cleaning:",
if False in np.isfinite(sm_img):
    print "Data could not be cleaned. Exiting..."
    exit(1)
print "Success"
print "_________________________________________________________"
# -------------------------------- Superpixel generation -------------------------------- #

width = sm_img.shape[0]
height = sm_img.shape[1]
dim = sm_img.shape[2]

# STEP1: Graph construction
edges = sb_spec.build_graph(run_flag, dist_flag, sm_img, ds_name)
print "Checkpoint : build graph ended."

# STEP2: Segment graph/Generate superpixels
univ = sb_spec.segment_graph(width*height, edges, k)
print "Checkpoint : segmentation ended."

# STEP 2: save_rgb
# contoured_img = extra.draw_contour(univ, sm_img, k, dist_flag)
# save_rgb('%s/step1_segmented.jpg' % output_dir, contoured_img, [1, 0, 2])

# STEP3: Post-processing segments/superpixels
# detect small components as the roads are very narrow. K and min_size controls the overall size
# and hence large k and min_size won't detect the thin line segments.
seg_id_px_arr = sb_spec.post_process(univ, edges, min_size, height, width)
print "Checkpoint : post-process 1 ended."

segmented_img = sb_spec.color_segments(seg_id_px_arr[:, 1], width, height)
save_rgb("%s/segmented_image_c_%s_sz_%s.jpg" % (output_dir, k, min_size), segmented_img, [0, 1, 2,])

# STEP 3: save_rgb
# contoured_img = extra.draw_contour(univ, sm_img, k, dist_flag)
# save_rgb('%s/step2_pp1.jpg' % output_dir, contoured_img, [1, 0, 2])

# STEP4: Calculate mean spectra of the segments/superpixels
seg_mean_spectra = sb_spec.get_mean_spectra(seg_id_px_arr, sm_img)
print "Checkpoint : calculating mean spectra ended."

# --------------------------------Superpixel generation ends-------------------------------- #

# --------------------------------Classification-------------------------------- #
# STEP5: Classify by nearest neighbor from the spectral library
classified_labels = sb_lib.classify_nn(seg_mean_spectra)
# classified_labels = sb_lib.classify_spectra_unmix(seg_mean_spectra)
print "Checkpoint : classification ended."

# Filter label roads (asphalt/concrete)
candidate_seg_id_px_arr = [seg_id_px_arr[i] for i, elem in enumerate(classified_labels) if elem == 1]
candidate_seg_id_px_arr = np.asarray(candidate_seg_id_px_arr)
print "# of candidate road segments: ", len(candidate_seg_id_px_arr)
print "_________________________________________________________"

# STEP 6: post process 2
# small k and min_size forces to detect small segments -> with low shape feature value.
# merge the connected segments that are small. then calculate shape feature value and filter shape
merged_candidate_seg_id_px_arr = sb_spec.post_process2(univ, edges, candidate_seg_id_px_arr, min_size2)
merged_candidate_seg_id_px_arr = merged_candidate_seg_id_px_arr[np.argsort(merged_candidate_seg_id_px_arr[:, 0])]
print "Checkpoint : post-process 2 ended."

# STEP 7: calculate LFI of segments
# lfi_list = sb_spec.get_lfi(merged_candidate_seg_id_px_arr, height, width)
shp_score_list, label_list = sb_spec.get_perim_area(merged_candidate_seg_id_px_arr, height, width)
print "Checkpoint : calculate shape score ended."

hist, bins = np.histogram(shp_score_list, 10)
w = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=w)
plt.savefig("%s/shape_score_hist_%s_%s" % (output_dir, k, min_size))
print "Checkpoint : shape score histogram generated."

# Test LFI
classified_img = np.zeros([width, height, 3], int)
red = np.array([255, 0, 0])

shp_score_img = np.zeros([width, height, 3], int)
col = np.empty([3], int)
for idx in range(len(shp_score_list) - 1, -1, -1):
    px_list = merged_candidate_seg_id_px_arr[idx][1]
    if 0 <= shp_score_list[idx] < 0.10:
        col = np.array([188, 143, 143]) # rose
    elif 0.10 <= shp_score_list[idx] < 0.20:
        col = np.array([255, 0, 255]) # magenta
    elif 0.20 <= shp_score_list[idx] < 0.30:
        col = np.array([255, 69, 0])  # orange
    elif 0.30 <= shp_score_list[idx] < 0.40:
        col = np.array([0, 255, 255])   # cyan
    elif 0.40 <= shp_score_list[idx] < 0.50:
        col = np.array([0, 255, 0]) # lime
    elif 0.50 <= shp_score_list[idx] < 0.60:
        col = np.array([255, 0, 0])   # red
    elif 0.6 <= shp_score_list[idx] < 0.70:
        col = np.array([0, 0, 255])  # blue
    elif 0.7 <= shp_score_list[idx] < 0.8:
        col = np.array([0, 128, 0])   # green
    elif 0.8 <= shp_score_list[idx] < 0.9:
        col = np.array([128, 0, 128])   # purple
    elif 0.9 <= shp_score_list[idx] < 1.0:
        col = np.array([255, 255, 0])  # chocolate
    elif shp_score_list[idx] >= 1.0:
        col = np.array([255, 255, 255])  # white

    for px in px_list:
        x = px % width
        y = px / width
        shp_score_img[x, y] = col
        classified_img[x, y] = red

save_rgb("%s/step3_classified_img_%s_%s_PP_%s.jpg" % (output_dir, k, min_size, min_size2), classified_img, [0, 1, 2,])
save_rgb("%s/step4_score_colors_%s_%s_PP_%s.jpg" % (output_dir, k, min_size, min_size2), shp_score_img, [0, 1, 2,])

# STEP8: Final filtering based on shape
road_seg_id_px_arr = sb_spec.filter_shape(merged_candidate_seg_id_px_arr, shp_score_list, label_list, 0, 0.6)
print "# of road segments after shape filtering: ", len(road_seg_id_px_arr)
print "Checkpoint : filter by shape ended."
print "_________________________________________________________"

# STEP7: Save classified road as output image
op_img = np.zeros((width, height, 3), int)
red = np.array([255, 0, 0])
road_pxs_lst = []
for i in range(0, len(road_seg_id_px_arr)):
        road_pxs_lst += road_seg_id_px_arr[i][1]
road_pxs = np.asarray(road_pxs_lst)

for px in road_pxs:
    op_img[px % width, px / width] = red

save_rgb("%s/Road_Final_Output_%s_SAD_%s_%s_PP2_%s.jpg" % (output_dir, ds_name, k, min_size, min_size2), op_img, [0, 1, 2,])
print "Output Road_Final_Output_%s_SAD_%s_%s_PP2_%s.jpg generated in %s/ " % (ds_name, k, min_size, min_size2, output_dir)
# STEP8: Compare with ground truth
precision_lst = []
recall_lst = []
precision, recall = sb_lib.calc_precision_recall(road_seg_id_px_arr, ds_name)

print "Precision: ", precision
print "Recall: ", recall

# saving output
recall_file = open("%s/recall_%s.txt" % (output_dir, min_size), 'a')
recall_file.write(str(recall))
recall_file.write("\n")
prec_file = open("%s/prec_%s.txt" % (output_dir, min_size), 'a')
prec_file.write(str(precision))
prec_file.write("\n")
recall_file.close()
prec_file.close()

