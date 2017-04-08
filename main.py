from spectral import *
import numpy as np
import cv2
import cPickle
import segment_spectra as sb_spec
import classification as sb_lib
import misc


# Parameter settings
run_flag = 'E'  # for existing run : E, for new run: N
dist_flag = 'SAD'  # SAD: SAD distance metric, EU: euclidean distance metric
sigma = 0.25
# c = int(sys.argv[1])
# min_size = int(sys.argv[2])
c = 70
min_size = 10

print "Parameters:"
print "c: ", c
print "Min_size: ", min_size
print "Distance metric: ", dist_flag

# -------------------------------- Load data -------------------------------- #

# Load whole image
# img = open_image("/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/HyMap01_Berlin_Urban_Gradient_2009.hdr")

# Load image subset
img = cPickle.load(open("data/hymap02_ds02.pkl", "rb"))

print "Image dimension:", img.shape
print "Data loaded"

# -------------------------------- Pre - processing -------------------------------- #

# Data cleaning
img = misc.remove_nan_inf(img)
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

# STEP1: Graph construction
edges = sb_spec.build_graph(run_flag, dist_flag, sm_img)

# STEP2: Segment graph/Generate superpixels
univ = sb_spec.segment_graph(width*height, edges, c)

# STEP3: Post-processing segments/superpixels
# detect small components as the roads are very thin. K and min_size controls the overall size
# and hence large k and min_size won't detect the thin line segments.
seg_id_px_arr, lfi_list = sb_spec.post_process(univ, edges, min_size, height, width)

# STEP4: Calculate mean spectra of the segments/superpixels
seg_mean_spectra = sb_spec.get_mean_spectra(seg_id_px_arr, sm_img)

# --------------------------------Superpixel generation ends-------------------------------- #

# --------------------------------Classification-------------------------------- #
# STEP5: Classify by nearest neighbor from the spectral library
classified_labels = sb_lib.classify(seg_mean_spectra)

# Filter label roads
asphalt_concrete_seg_id_px_arr = [seg_id_px_arr[i] for i, elem in enumerate(classified_labels) if elem == 1]

# STEp 6: post process 2

# small k and min_size forces to detect small segments -> with low LFI.
# merge the connected segments that are small. then calculate lfi and filter shape



# STEP 6: filter_asph_con_edge
# STEP 7: find asph con segment edges
# STEP 8: merge small segments
filtered_lfi_list = [lfi_list[i] for i, elem in enumerate(classified_labels) if elem == 1]




# STEP6: Final filtering based on shape
road_seg_px_list = sb_spec.filter_shape(asphalt_concrete_seg_id_px_arr, filtered_lfi_list, 0, 5)

# STEP7: Compare with ground truth
precision_lst = []
recall_lst = []
precision, recall, road_pxs = sb_lib.calc_precision_recall(asphalt_concrete_seg_px_arr)

print "Precision: ", precision
print "Recall: ", recall

# STEP8: Save classified road as output image
op_img = sm_img[:, :, 39:42]
red = np.array([0, 0, 254])

for px in road_pxs:
    op_img[px % width, px / width] = red

cv2.imwrite("output/shell output/LFI/road_op_hymap02ds02_SAD_mean_%s_%s_NEW.jpg" % (c, min_size), op_img)