from spectral import *
import numpy as np
import cv2
import cPickle
import segment_spectra as sb_spec
import classification as sb_lib
import misc
import matplotlib.pyplot as plt
import extra


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
    gt = cv2.imread("data/%s_infra_GT_highroads.jpg" % ds_name)
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
        #find segment id of the pixel
        # seg_index= seg_id_px_arr[:,0].index(univ.find(fn_px)) #segid_list_u.index(u.find(fn_px))
        seg_index = np.where(seg_id_px_arr[:,0]==univ.find(fn_px))[0][0] 
        #find label for the segment id
        labels_level2.append(classified_labels[seg_index])
        #find abundance vector of the segment
        ab = abundance[:,seg_index]
        labels1_level_spectra.append(np.argmax(ab))
        #find count of segment mapped to spectra 57        
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
input_dir = "/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/HyMap02_Berlin_Urban_Gradient_2009.hdr"
output_dir = "output/shell output/perim_area/ds02"

print "Parameters:"
print "c: ", c
print "Min_size: ", min_size
print "Distance metric: ", dist_flag

# -------------------------------- Load data -------------------------------- #

# Load whole image
# img = open_image(input_dir)

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

# STEP 2: save_rgb
# contoured_img = extra.draw_contour(univ, sm_img, c, dist_flag)
# save_rgb('%s/step1_segmented.jpg' % output_dir, contoured_img, [1, 0, 2])

# STEP3: Post-processing segments/superpixels
# detect small components as the roads are very narrow. K and min_size controls the overall size
# and hence large k and min_size won't detect the thin line segments.
seg_id_px_arr = sb_spec.post_process(univ, edges, min_size, height, width)
print "Checkpoint : post-process 1 ended."

# segmented_img = sb_spec.color_segments(seg_id_px_arr[:, 1], width, height)
# cv2.imwrite("%s/segmented_image_c_%s_sz_%s.jpg" % (output_dir, c, min_size), segmented_img)

# STEP 3: save_rgb
# contoured_img = extra.draw_contour(univ, sm_img, c, dist_flag)
# save_rgb('%s/step2_pp1.jpg' % output_dir, contoured_img, [1, 0, 2])

# STEP4: Calculate mean spectra of the segments/superpixels
seg_mean_spectra = sb_spec.get_mean_spectra(seg_id_px_arr, sm_img)
print "Checkpoint : mean spectra ended."

# --------------------------------Superpixel generation ends-------------------------------- #

# --------------------------------Classification-------------------------------- #
# STEP5: Classify by nearest neighbor from the spectral library
# classified_labels = sb_lib.classify_nn(seg_mean_spectra)
abundance = sb_lib.unmix_spectra(seg_mean_spectra)
classified_labels = sb_lib.classify_spectra_unmix(seg_mean_spectra)
print "Checkpoint : classification ended."

# Filter label roads (asphalt/concrete)
candidate_seg_id_px_arr = [seg_id_px_arr[i] for i, elem in enumerate(classified_labels) if elem == 1]
candidate_seg_id_px_arr = np.asarray(candidate_seg_id_px_arr)
print "# of segments labelled as road: ", len(candidate_seg_id_px_arr)
print "Checkpoint : filter by road label ended."

# STEP 6: post process 2
# small k and min_size forces to detect small segments -> with low shape feature value.
# merge the connected segments that are small. then calculate shape feature value and filter shape
# merged_candidate_seg_id_px_arr = sb_spec.post_process2(univ, edges, candidate_seg_id_px_arr, 30)
# merged_candidate_seg_id_px_arr = candidate_seg_id_px_arr
# merged_candidate_seg_id_px_arr = merged_candidate_seg_id_px_arr[np.argsort(merged_candidate_seg_id_px_arr[:, 0])]
# print "Checkpoint : post-process 2 ended."


# color_merged_segs = sb_spec.color_segments(merged_candidate_seg_id_px_arr[:, 1], width, height)
# cv2.imwrite("output/shell output/LFI/post_process2_op_0.png", color_merged_segs)

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
for i in range(0, len(road_seg_id_px_arr)):
        road_pxs_lst += road_seg_id_px_arr[i][1]
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
    #seg_id=segid_list_u.index(u.find(376*width+x))
    ab = abundance[:,seg_index]
    if np.argmax(ab) == 57:
        print x





