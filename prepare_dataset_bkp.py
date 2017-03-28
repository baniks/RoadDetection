from spectral import *
import numpy as np
import cPickle

img = open_image("/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/EnMAP01_Berlin_Urban_Gradient_2009.hdr")
####################################################################################
# Dataset1
part1 = img[32:35, 90:93, :]
part2 = img[36:51, 90:93, :]
part3 = img[52:88, 90:93, :]
part4 = img[90:94, 90:93, :]
part5 = img[99:110, 90:93, :]
dataset1_part1 = np.concatenate((part1, part2, part3, part4, part5),  axis=0)
part1 = img[32:35, 94:130, :]
part2 = img[36:51, 94:130, :]
part3 = img[52:88, 94:130, :]
part4 = img[90:94, 94:130, :]
part5 = img[99:110, 94:130, :]
dataset1_part2 = np.concatenate((part1, part2, part3, part4, part5),  axis=0)
dataset1 = np.concatenate((dataset1_part1, dataset1_part2), axis=1)
# #clean band
# part1 = dataset1_allbands[:, :, 0:23]
# part2 = dataset1_allbands[:, :, 38:85]
# part3 = dataset1_allbands[:, :, 86:]
# dataset1 = np.concatenate((part1,  part2,  part3),  axis=2)

view = imshow(dataset1, bands=(4, 17, 52, ))
cPickle.dump(dataset1, open("data/dataset1.pkl", "wb"))


#####################################################################################
# Dataset2
part1=img[115:164, 90:93, :]
part2=img[165:179, 90:93, :]
dataset2_part1 = np.concatenate((part1, part2, ),  axis=0)
part1=img[115:164, 94:130, :]
part2=img[165:179, 94:130, :]
dataset2_part2 = np.concatenate((part1, part2, ),  axis=0)
dataset2 = np.concatenate((dataset2_part1, dataset2_part2), axis=1)
# # clean band
# part1 = dataset2_allbands[:, :, 0:23]
# part2 = dataset2_allbands[:, :, 38:85]
# part3 = dataset2_allbands[:, :, 86:]
# dataset2 = np.concatenate((part1,  part2,  part3),  axis=2)

view= imshow(dataset2, bands=(4, 17, 52, ))
cPickle.dump(dataset2, open("data/dataset2.pkl", "wb"))

####################################################################################
# Dataset3
part1 = img[295:305, 90:93, :]
part2 = img[307:313, 90:93, :]
part3 = img[314:344, 90:93, :]
part4 = img[346:352, 90:93, :]
part5 = img[353:358, 90:93, :]
part6 = img[359:361, 90:93, :]
dataset3_part1 = np.concatenate((part1, part2, part3, part4, part5, part6),  axis=0)

part1 = img[295:305, 94:130, :]
part2 = img[307:313, 94:130, :]
part3 = img[314:344, 94:130, :]
part4 = img[346:352, 94:130, :]
part5 = img[353:358, 94:130, :]
part6 = img[359:361, 94:130, :]
dataset3_part2 = np.concatenate((part1, part2, part3, part4, part5, part6),  axis=0)
dataset3 = np.concatenate((dataset3_part1, dataset3_part2), axis=1)
# # clean band
# part1 = dataset3_allbands[:, :, 0:23]
# part2 = dataset3_allbands[:, :, 38:85]
# part3 = dataset3_allbands[:, :, 86:]
# dataset3 = np.concatenate((part1,  part2,  part3),  axis=2)
view = imshow(dataset3, bands=(4, 17, 52, ))
cPickle.dump(dataset3, open("data/dataset3.pkl", "wb"))

####################################################################################
# Ground truth

gt_level1 = open_image(
    "/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 02 additional data/02_additional_data/land_cover/LandCov_Layer_Level2_Berlin_Urban_Gradient_2009.hdr")
part1 = gt_level2[115:164, 90:93, :]
part2 = gt_level2[165:179, 90:93, :]
gt_dataset2_part1 = np.concatenate((part1, part2, ),  axis=0)
part1 = gt_level2[115:164, 94:130, :]
part2 = gt_level2[165:179, 94:130, :]
gt_dataset2_part2 = np.concatenate((part1, part2, ),  axis=0)
gt_dataset2 = np.concatenate((gt_dataset2_part1, gt_dataset2_part2), axis=1)
# # clean band
# part1 = dataset2_allbands[:, :, 0:23]
# part2 = dataset2_allbands[:, :, 38:85]
# part3 = dataset2_allbands[:, :, 86:]
# dataset2 = np.concatenate((part1,  part2,  part3),  axis=2)

view= imshow(gt_dataset2, bands=(2, ))
cPickle.dump(gt_dataset2, open("data/gt_dataset2.pkl", "wb"))
