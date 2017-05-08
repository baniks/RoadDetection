#!/usr/bin/python
#######################################################################
#   File name: analyze_unmix_op.py
#   Author: Soubarna Banik
#   Description: ad-hoc script for analysing spectral unmixing output
#######################################################################

from spectral import *
import numpy as np
import cv2
import cPickle
import segment_spectra as sb_spec
import classification as sb_lib
import matplotlib.pyplot as plt


def calc_precision_recall(road_seg_id_px_list, abundance, classified_labels, seg_id_px_arr, ds_name):
    """
    Calculates precision and recall by comparing the pixels labelled as road with ground truth
    :param road_seg_id_px_list: segments mapped as roads and corresponding segment id to pixel mapping
    :param ds_name: dataset name
    :return:
    precision: calculated precision value
    recall: calculated recall value
    """
    # Load GT (infrared image, roads marked with blue manually)
    gt = cv2.imread("../data/%s_infra_GT_highroads.jpg" % ds_name)

    # Find pixels mapped as road (marked in blue) from ground truth image
    blue = np.array([254, 0, 0])
    gt_road_pxs_lst = []
    w = gt.shape[0]
    h = gt.shape[1]

    for x in range(0, w):
        for y in range(0, h):
            if np.array_equal(gt[x, y], blue):
                # gt_road_pxs_lst.append(np.array([x, y]))
                gt_road_pxs_lst.append(y * w + x)

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
    fn_set = []

    if len(road_seg_id_px_list) > 0:
        tp = len(np.intersect1d(gt_road_pxs, road_pxs))
        fp = len(np.setdiff1d(road_pxs, gt_road_pxs))
        fn = len(np.setdiff1d(gt_road_pxs, road_pxs))
        fn_set = np.setdiff1d(gt_road_pxs, road_pxs)
        precision = tp/(float(tp) + fp)
        recall = tp / (float(tp) + fn)

    print "tp, fp, fn: ", tp, fp, fn

    cnt=0
    labels_level2=[]
    labels1_level_spectra=[]

    for fn_px in fn_set:
        # find segment id of the pixel
        # seg_index= seg_id_px_arr[:,0].index(univ.find(fn_px)) #segid_list_u.index(u.find(fn_px))
        seg_index = np.where(seg_id_px_arr[:,0]==univ.find(fn_px))[0][0] 

        # find label for the segment id
        labels_level2.append(classified_labels[seg_index])

        # find abundance vector of the segment
        ab = abundance[:,seg_index]
        labels1_level_spectra.append(np.argmax(ab))

        # find count of segment mapped to spectra 57
        if np.argmax(ab) == 57:
            cnt+=1

    return precision, recall, road_pxs, labels_level2, labels1_level_spectra


# Parameter settings
run_flag = 'E'  # for existing run : E, for new run: N
dist_flag = 'SAD'  # SAD: SAD distance metric, EU: euclidean distance metric
sigma = 0.25
#c = int(sys.argv[1])
#min_size = int(sys.argv[2])
#ds_name = str(sys.argv[3])

c = 100
min_size = 10
ds_name = "hymap02_ds02"
input_dir = "../data/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/HyMap02_Berlin_Urban_Gradient_2009.hdr"
output_dir = "../output/shell output/perim_area/ds02"

print "Parameters:"
print "c: ", c
print "Min_size: ", min_size
print "Distance metric: ", dist_flag

# -------------------------------- Load data -------------------------------- #

# Load whole image
# img = open_image(input_dir)

# Load image subset
img = cPickle.load(open("../data/%s.pkl" % ds_name, "rb"))

# Load spectral library
lib = open_image("../data/Hyperspectral data/Berlin Urban Gradient 2009 02 additional data/02_additional_data/spectral_library/SpecLib_Berlin_Urban_Gradient_2009.hdr")
spectra_lib = lib.spectra

print "Image dimension:", img.shape
print "Data loaded"

# -------------------------------- Pre - processing -------------------------------- #

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
# detect small components as the roads are very narrow. K and min_size controls the overall size
# and hence large k and min_size won't detect the thin line segments.
seg_id_px_arr = sb_spec.post_process(univ, edges, min_size, height, width)
print "Checkpoint : post-process 1 ended."

# STEP4: Calculate mean spectra of the segments/superpixels
seg_mean_spectra = sb_spec.get_mean_spectra(seg_id_px_arr, sm_img)
print "Checkpoint : mean spectra ended."

# --------------------------------Superpixel generation ends-------------------------------- #

# --------------------------------Classification-------------------------------- #
# STEP5: Classify by nearest neighbor from the spectral library
abundance = sb_lib.unmix_spectra(seg_mean_spectra)
classified_labels = sb_lib.classify_spectra_unmix(seg_mean_spectra)
print "Checkpoint : classification ended."

# Filter label roads (asphalt/concrete)
candidate_seg_id_px_arr = [seg_id_px_arr[i] for i, elem in enumerate(classified_labels) if elem == 1]
candidate_seg_id_px_arr = np.asarray(candidate_seg_id_px_arr)
print "# of segments labelled as road: ", len(candidate_seg_id_px_arr)
print "Checkpoint : filter by road label ended."

# STEP7: Compare with ground truth
precision_lst = []
recall_lst = []
precision, recall, road_pxs,labels_level2, labels1_level_spectra = calc_precision_recall(candidate_seg_id_px_arr, abundance, classified_labels, seg_id_px_arr, ds_name)

print "Precision: ", precision
print "Recall: ", recall
# Precision:  0.139344262295
# Recall:  0.0428211586902

# tp, fp, fn:  68 420 1520
for a in np.unique(labels1_level_spectra):
    print labels1_level_spectra.count(a)

# array([37, 50, 54, 57, 58, 60, 62, 69])
#>>> labels1_level_spectra.count(37) #2
#>>> labels1_level_spectra.count(50) #1
#>>> labels1_level_spectra.count(54) #31
#>>> labels1_level_spectra.count(57) #1174
#>>> labels1_level_spectra.count(58) #3
#>>> labels1_level_spectra.count(60) #301
#>>> labels1_level_spectra.count(62) #2
#>>> labels1_level_spectra.count(69) #6

# false negative break up
for a in np.unique(labels_level2):
    print labels_level2.count(a)
#array([2, 3, 4, 5])
#>>> labels_level2.count(2) #2
#>>> labels_level2.count(3) #1510
#>>> labels_level2.count(4) #2
#>>> labels_level2.count(5) #6

# false negative pixel = 11270 (seg_id:15270) classified as spectral 57
x = 11270% width
y = 11270/ width
px = sm_img[x, y]
tree_px=sm_img[267,376]
fig = plt.figure()
fig.hold(1)
ax = fig.add_subplot(111)
ax.set_xlim(0.40,2.5)
wv = np.array([0.455400, 0.469400, 0.484300, 0.499100, 0.513800, 0.528800, 0.543600,
 0.558400, 0.573100, 0.588100, 0.602900, 0.617600, 0.632000, 0.646500,
 0.660900, 0.675500, 0.690000, 0.704500, 0.718900, 0.733200, 0.747600,
 0.761800, 0.775900, 0.790100, 0.804600, 0.818800, 0.832900, 0.847100,
 0.861100, 0.874700, 0.887800, 0.893000, 0.908500, 0.923900, 0.939400,
 0.955200, 0.970400, 0.985800, 1.001400, 1.016600, 1.031800, 1.046900,
 1.062000, 1.076600, 1.091300, 1.106200, 1.120800, 1.135300, 1.149700,
 1.164100, 1.178600, 1.192800, 1.206900, 1.221000, 1.235100, 1.249200,
 1.263100, 1.277000, 1.290700, 1.304300, 1.318300, 1.330100, 1.505000,
 1.518800, 1.532600, 1.546300, 1.559800, 1.573200, 1.586400, 1.599500,
 1.612700, 1.625900, 1.638900, 1.651700, 1.664400, 1.677100, 1.689600,
 1.702100, 1.714600, 1.726900, 1.739300, 1.751500, 1.763600, 1.775600,
 1.787600, 1.798100, 2.027500, 2.046700, 2.065500, 2.084100, 2.102500,
 2.120900, 2.139000, 2.157000, 2.174700, 2.191700, 2.210300, 2.228100,
 2.245600, 2.263400, 2.280400, 2.297400, 2.314400, 2.331400, 2.348300,
 2.365000, 2.381500, 2.397700, 2.414100, 2.430300, 2.446500])

plt_px,=ax.plot(wv,px,'r--',linewidth=3.0,label='Pixel (70,28)')
plt_tree_px,=ax.plot(wv,tree_px,'b--',linewidth=3.0,label='Pixel (267,376)')
plt_c1,=ax.plot(wv,spectra_lib[23],'#CC0000',linewidth=3.0,label='Ashphalt1')
plt_c2,=ax.plot(wv,spectra_lib[24],'#CC0066',linewidth=3.0,label='Ashphalt2')
plt_c3,=ax.plot(wv,spectra_lib[25],'#990000',linewidth=3.0,label='Ashphalt3')
plt_c4,=ax.plot(wv,spectra_lib[26],'#FF8000',linewidth=3.0,label='Ashphalt4')
plt_c5,=ax.plot(wv,spectra_lib[27],'#000099',linewidth=3.0,label='Concrete1')
plt_c6,=ax.plot(wv,spectra_lib[28],'#660066',linewidth=3.0,label='Concrete2')
plt_c7,=ax.plot(wv,spectra_lib[29],'#009999',linewidth=3.0,label='Concrete3')
plt_g,=ax.plot(wv,spectra_lib[57],'g--',linewidth=3.0,label='Deciduous tree 10')
plt.legend(handles=[plt_px,plt_tree_px,plt_c1,plt_c2,plt_c3,plt_c4,plt_c5,plt_c6,plt_c7,plt_g])
leg = plt.gca().get_legend()
ltext  = leg.get_texts()  
llines = leg.get_lines() 
plt.setp(ltext, fontsize=18)    # the legend text fontsize
plt.setp(llines, linewidth=2.5)
plt.ylabel('Reflectance',fontsize=18)
plt.xlabel('Wavelength(micrometer)',fontsize=18)
plt.show()


# STEP8: Save classified road as output image
op_img = np.zeros((width, height, 3), int)
red = np.array([0, 0, 255])
road_pxs_lst = []
for i in range(0, len(candidate_seg_id_px_arr)):
        road_pxs_lst += candidate_seg_id_px_arr[i][1]
road_pxs = np.asarray(road_pxs_lst)

for px in road_pxs:
    op_img[px % width, px / width] = red

cv2.imwrite("%s/step5_road_op_%s_SAD_%s_%s_PP2_30.jpg" % (output_dir, ds_name, c, min_size), op_img)

# saving output
recall_file = open("%s/recall_%s.txt" % (output_dir, min_size), 'a')
recall_file.write(str(recall))
recall_file.write("\n")
prec_file = open("%s/prec_%s.txt" % (output_dir, min_size), 'a')
prec_file.write(str(precision))
prec_file.write("\n")
recall_file.close()
prec_file.close()


for x in range(196,300):
    seg_index = np.where(seg_id_px_arr[:,0]==univ.find(376*width+x))[0][0] 
    ab = abundance[:,seg_index]
    if np.argmax(ab) == 57:
        print x





