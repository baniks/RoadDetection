from spectral import *
import cPickle
import numpy as np
import sys


# Scene: hymap02
hymap02 = open_image("/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/HyMap02_Berlin_Urban_Gradient_2009.hdr")

# preparing two subsets - dataset 1 and 2
# hymap02_ds01 = hymap02[150:500, 90:700, :]
hymap02_ds01 = hymap02[1500:2000, 50:630, :]
hymap02_ds02 = hymap02[2000:2400, 40:630, :]

# display
# view1 = imshow(hymap02_ds01, bands=(18, 5, 1,))
# view2 = imshow(hymap02_ds02, bands=(18, 5, 1,))

# saving dataset
cPickle.dump(hymap02_ds01, open("data/hymap02_ds01.pkl", "wb"))
cPickle.dump(hymap02_ds02, open("data/hymap02_ds02.pkl", "wb"))
save_rgb('images/hymap02/ds01/hymap02_ds01.jpg', hymap02_ds01, [18, 5, 1])
save_rgb('images/hymap02/ds01/hymap02_ds01_27_74_13.jpg', hymap02_ds01, [27, 74, 13])
save_rgb('images/hymap02/ds01/hymap02_ds01_grey.jpg', hymap02_ds01, [5,])
save_rgb('images/hymap02/ds01/hymap02_ds01_infra.jpg', hymap02_ds01, [39,])

save_rgb('images/hymap02/ds02/hymap02_ds02.jpg', hymap02_ds02, [18, 5, 1])
save_rgb('images/hymap02/ds02/hymap02_ds02_27_74_13.jpg', hymap02_ds02, [27, 74, 13])
save_rgb('images/hymap02/ds02/hymap02_ds02_grey.jpg', hymap02_ds02, [5,])
save_rgb('images/hymap02/ds02/hymap02_ds02_infra.jpg', hymap02_ds02, [39,])
	
# Scene: hymap01
hymap01 = open_image("/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 01 image products/01_image_products/HyMap01_Berlin_Urban_Gradient_2009.hdr")
hymap01_ds01 = hymap01[3750:5000, 530:1150, :]
hymap01_ds02 = hymap01[5000:6000, 530:1150, :]

# view1 = imshow(hymap01_ds01, bands=(21, 74, 13,))
# removing corrupt bands
part1 = hymap01_ds01[:, :, 0:22]
part2 = hymap01_ds01[:, :, 38:83]
part3 = hymap01_ds01[:, :, 85:110]
hymap01_ds01_e = np.concatenate((part1, part2, part3), axis=2)

part1 = hymap01_ds02[:, :, 0:22]
part2 = hymap01_ds02[:, :, 38:83]
part3 = hymap01_ds02[:, :, 85:110]
hymap01_ds02_e = np.concatenate((part1, part2, part3), axis=2)

# saving dataset
cPickle.dump(hymap01_ds01_e, open("data/hymap01_ds01.pkl", "wb"))
cPickle.dump(hymap01_ds02_e, open("data/hymap01_ds02.pkl", "wb"))
save_rgb('images/hymap01/hymap01_ds01.jpg', hymap01_ds01_e, [18, 5, 1])
save_rgb('images/hymap01/hymap01_ds01_21_74_13.jpg', hymap01_ds01, [21, 74, 13])
save_rgb('images/hymap01/hymap01_ds02.jpg', hymap01_ds02_e, [18, 5, 1])
save_rgb('images/hymap01/hymap01_ds02_21_74_13.jpg', hymap01_ds02, [21, 74, 13])
