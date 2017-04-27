lib = open_image("/home/soubarna/Documents/Independent Study/Hyperspectral data/Berlin Urban Gradient 2009 02 additional data/02_additional_data/spectral_library/SpecLib_Berlin_Urban_Gradient_2009.hdr")
spectra_lib=lib.spectra
wv=np.array([0.455400, 0.469400, 0.484300, 0.499100, 0.513800, 0.528800, 0.543600,
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

#misclassfied pixel spectra
#px = 203,227

px = sm_img[203,227]
seg_id=segid_list_u.index(u.find(227*width+203)) #2571
ab = abundance_fa[:,seg_id]
np.argmax(ab) #57

plt.figure()
plt.hold(1)
plt_px,=plt.plot(wv,px,'k--',linewidth=2.0,label='Pixel (203,227)')
plt_c1,=plt.plot(wv,spectra_lib[23],'b',linewidth=2.0,label='Ashphalt1')
plt_c2,=plt.plot(wv,spectra_lib[24],'g',linewidth=2.0,label='Ashphalt2')
plt_c3,=plt.plot(wv,spectra_lib[25],'c',linewidth=2.0,label='Ashphalt3')
plt_c4,=plt.plot(wv,spectra_lib[26],'y',linewidth=2.0,label='Ashphalt4')
plt_c5,=plt.plot(wv,spectra_lib[27],'m',linewidth=2.0,label='Concrete1')
plt_c6,=plt.plot(wv,spectra_lib[28],'k',linewidth=2.0,label='Concrete2')
plt_c7,=plt.plot(wv,spectra_lib[29],'r',linewidth=2.0,label='Concrete3')
plt_g,=plt.plot(wv,spectra_lib[57],'g--',linewidth=2.0,label='Deciduous tree 10')
plt.legend(handles=[plt_px,plt_c1,plt_c2,plt_c3,plt_c4,plt_c5,plt_c6,plt_c7,plt_g])
plt.ylabel('Reflectance')
plt.xlabel('Wavelength(micrometer)')
plt.show()


#Px = 77,90
px = sm_img[77,90]

seg_id=segid_list_u.index(u.find(90*width+77)) #1011
ab = abundance_fa[:,seg_id]
np.argmax(ab) #57

plt.figure()
plt.hold(1)
plt_px,=plt.plot(wv,px,'k--',linewidth=2.0,label='Pixel (77,90)')
plt_c1,=plt.plot(wv,spectra_lib[23],'b',linewidth=2.0,label='Ashphalt1')
plt_c2,=plt.plot(wv,spectra_lib[24],'g',linewidth=2.0,label='Ashphalt2')
plt_c3,=plt.plot(wv,spectra_lib[25],'c',linewidth=2.0,label='Ashphalt3')
plt_c4,=plt.plot(wv,spectra_lib[26],'y',linewidth=2.0,label='Ashphalt4')
plt_c5,=plt.plot(wv,spectra_lib[27],'m',linewidth=2.0,label='Concrete1')
plt_c6,=plt.plot(wv,spectra_lib[28],'k',linewidth=2.0,label='Concrete2')
plt_c7,=plt.plot(wv,spectra_lib[29],'r',linewidth=2.0,label='Concrete3')
plt_g,=plt.plot(wv,spectra_lib[57],'g--',linewidth=2.0,label='Deciduous tree 10')
plt.legend(handles=[plt_px,plt_c1,plt_c2,plt_c3,plt_c4,plt_c5,plt_c6,plt_c7,plt_g])
plt.ylabel('Reflectance')
plt.xlabel('Wavelength(micrometer)')
plt.show()

# correct classified px spectra
# px=150 243
px=sm_img[150,243]
seg_id=segid_list_u.index(u.find(243*width+150)) #2814
ab = abundance_fa[:,seg_id]
np.argmax(ab) #27

plt.figure()
plt.hold(1)
plt_px,=plt.plot(wv,px,'k--',linewidth=2.0,label='Pixel (150,243)')
plt_c1,=plt.plot(wv,spectra_lib[23],'b',linewidth=2.0,label='Ashphalt1')
plt_c2,=plt.plot(wv,spectra_lib[24],'g',linewidth=2.0,label='Ashphalt2')
plt_c3,=plt.plot(wv,spectra_lib[25],'c',linewidth=2.0,label='Ashphalt3')
plt_c4,=plt.plot(wv,spectra_lib[26],'y',linewidth=2.0,label='Ashphalt4')
plt_c5,=plt.plot(wv,spectra_lib[27],'m',linewidth=2.0,label='Concrete1')
plt_c6,=plt.plot(wv,spectra_lib[28],'k',linewidth=2.0,label='Concrete2')
plt_c7,=plt.plot(wv,spectra_lib[29],'r',linewidth=2.0,label='Concrete3')
plt_g,=plt.plot(wv,spectra_lib[57],'g--',linewidth=2.0,label='Deciduous tree 10')
plt.legend(handles=[plt_px,plt_c1,plt_c2,plt_c3,plt_c4,plt_c5,plt_c6,plt_c7,plt_g])
plt.ylabel('Reflectance')
plt.xlabel('Wavelength(micrometer)')
plt.show()

#True tree 57 pixel
196-251
tree->228*width+385
78785 -100785
for y in range(196,251):
    seg_id=segid_list_u.index(u.find(y*width+385))
    ab = abundance_fa[:,seg_id]
    if np.argmax(ab) == 57:
        print y

output: 204,205,206
px=sm_img[385,204]
plt.figure()
plt.hold(1)
plt_px,=plt.plot(wv,px,'k--',linewidth=2.0,label='Pixel (385,204)')
plt_c1,=plt.plot(wv,spectra_lib[23],'b',linewidth=2.0,label='Ashphalt1')
plt_c2,=plt.plot(wv,spectra_lib[24],'g',linewidth=2.0,label='Ashphalt2')
plt_c3,=plt.plot(wv,spectra_lib[25],'c',linewidth=2.0,label='Ashphalt3')
plt_c4,=plt.plot(wv,spectra_lib[26],'y',linewidth=2.0,label='Ashphalt4')
plt_c5,=plt.plot(wv,spectra_lib[27],'m',linewidth=2.0,label='Concrete1')
plt_c6,=plt.plot(wv,spectra_lib[28],'k',linewidth=2.0,label='Concrete2')
plt_c7,=plt.plot(wv,spectra_lib[29],'r',linewidth=2.0,label='Concrete3')
plt_g,=plt.plot(wv,spectra_lib[57],'g--',linewidth=2.0,label='Deciduous tree 10')
plt.legend(handles=[plt_px,plt_c1,plt_c2,plt_c3,plt_c4,plt_c5,plt_c6,plt_c7,plt_g])
plt.ylabel('Reflectance')
plt.xlabel('Wavelength(micrometer)')
plt.show()


#identifying false_negative's classified labels
seg_px_list = sb_spec.get_segment_to_pxs(u, height, width, segid_list_u)
pavement_pxs = [seg_px_list[i] for i, elem in enumerate(classified_labels) if elem == 1]

pavement_pxs_lst = []
for i in range(0,len(pavement_pxs)):
    pavement_pxs_lst+=pavement_pxs[i]

pavement_pxs = np.asarray(pavement_pxs_lst)


gt = cv2.imread("images/hymap02_ds03_infra_E_highroads.jpg")

blue = np.array([254, 0, 0])
gt_road_pxs_lst = []
for x in range(0, gt.shape[0]):
    for y in range(0, gt.shape[1]):
        if np.array_equal(gt[x, y], blue):
            gt_road_pxs_lst.append(np.array([x,y]))
gt_road_pxs = np.asarray(gt_road_pxs_lst)


fn = len(sb_lib.multidim_difference(gt_road_pxs, pavement_pxs)) #1575
fn_set=sb_lib.multidim_difference(gt_road_pxs, pavement_pxs)


cnt=0
labels=[]
labels1=[]
for fn_px in fn_set:
    seg_id=segid_list_u.index(u.find(fn_px[0]*width+fn_px[1])) 
    labels.append(classified_labels[seg_id])
    ab = abundance_fa[:,seg_id]
    labels1.append(np.argmax(ab))
    if np.argmax(ab) == 57:
        cnt+=1

# 57 => 1330
>>> np.histogram(labels1)
(array([ 143,    0,    0,    0,    0,    0,    0,   15, 1332,   85]), array([ 35. ,  37.7,  40.4,  43.1,  45.8,  48.5,  51.2,  53.9,  56.6,
        59.3,  62. ]))
>>> labels1.count(35) #6
>>> labels1.count(37) #137
>>> labels1.count(54) #15
>>> labels1.count(57) #1330
>>> labels1.count(58) #2
>>> labels1.count(60) #57
>>> labels1.count(62) #28
-------------------------
# Total 1575
>>> np.histogram(labels)
(array([ 143,    0,    0,    0,    0, 1404,    0,    0,    0,   28]), array([ 2. ,  2.2,  2.4,  2.6,  2.8,  3. ,  3.2,  3.4,  3.6,  3.8,  4. ]))


#False negative break up
>>> labels.count(2) #143 low veg
>>> labels.count(3) #1404 tree
>>> labels.count(4) #28 soil
-------------------------------
#Total 1575

np.argmax(ab) == 37 => px = 181,192
px=sm_img[181,192]
plt.figure()
plt.hold(1)
plt_px,=plt.plot(wv,px,'k--',linewidth=2.0,label='Pixel (116,161)')
plt_c1,=plt.plot(wv,spectra_lib[23],'b',linewidth=2.0,label='Ashphalt1')
plt_c2,=plt.plot(wv,spectra_lib[24],'g',linewidth=2.0,label='Ashphalt2')
plt_c3,=plt.plot(wv,spectra_lib[25],'c',linewidth=2.0,label='Ashphalt3')
plt_c4,=plt.plot(wv,spectra_lib[26],'y',linewidth=2.0,label='Ashphalt4')
plt_c5,=plt.plot(wv,spectra_lib[27],'m',linewidth=2.0,label='Concrete1')
plt_c6,=plt.plot(wv,spectra_lib[28],'k',linewidth=2.0,label='Concrete2')
plt_c7,=plt.plot(wv,spectra_lib[29],'r',linewidth=2.0,label='Concrete3')
plt_g,=plt.plot(wv,spectra_lib[37],'g--',linewidth=2.0,label='Grass2')
plt.legend(handles=[plt_px,plt_c1,plt_c2,plt_c3,plt_c4,plt_c5,plt_c6,plt_c7,plt_g])
plt.ylabel('Reflectance')
plt.xlabel('Wavelength(micrometer)')
plt.show()

px=sm_img[116,161]
def diffSAD(sp1,sp2):
    numr = 0.0
    denom_1 = 0.0
    denom_2 = 0.0
    px_1 = sp1
    px_2 = sp2
    for d in range(0, sp1.shape[0]):
        numr += px_1[d]*px_2[d]
        denom_1 += px_1[d] * px_1[d]
        denom_2 += px_2[d] * px_2[d]
    val = math.acos(numr / (math.sqrt(denom_1) * math.sqrt(denom_2)))
    return val
