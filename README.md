# Superpixel-base Road detection using Hyperspectral Image #
A superpixel based unsupervised road detection system, using remotely sensed hyperspectral images is implemented.
The model is computationally efficient as it operates on superpixels rather than individual pixels. The use of hy-
perspectral images helps to identify roads accurately by utilizing the knowledge of the reflectance spectra of materials used in construction of roads. The model aims to eliminate the dependency on human intervention, and follows an unsupervised
approach. The program uses hyperspectral images and a spectral library available from the Berlin-Urban-Gradient dataset (http://pmd.gfz-potsdam.de/enmap/showshort.php?id=escidoc:1480925). The model needs three input parameters for superpixel segmentation - k, min_size and min_size2 and generates an output image with labelled road segments in the output directory.

## Dependency ##
* Python: 2.6+  
* List of libraries used:
  * numpy, scipy, matplotlib, math
  * for reading hyperspectral image: spectral (pip install spectral [http://www.spectralpython.net/installation.html])
  * for image processing: skimage (pip install scikit-image [http://scikit-image.org/docs/dev/install.html])
  * for storing pre-processed data: cPickle (pip install cPickle [https://pypi.python.org/pypi/cpickle/0.1])  
  * for mapping shapefile polygon to pixel/rasterization (optional): ogr, gdal

## Configurations Settings ##
A couple of threshold parameters need to be set for running the program. These are configured in main.py.  
**Threshold Parameters - Default value:**  
* k = 100
* min_size = 10
* min_size = 20  

**Other Parameters:** 
* run_flag: 'N' for fresh run (default), 'E' for loading existing graph file from data directory
* dist_flag: 'SAD' for spectral angle distance metric (default), 'EU' for euclidean distance metric
* sigma: standard deviation used for gaussian smoothing. Default value is 0.25
* ds_name: input dataset name. This is used through out the program for loading input data file, ground truth data and for generating output file.

## Directory structure/Input files ##
output_dir: \<PATH_TO_OUTPUT_DIRECTORY\>  
data_dir: \<PATH_TO_PREPROCESSED_SUBSET_DATA\>  
input spectral file: \<ds_name\>.pkl   
input ground truth file: \<ds_name\>_infra_GT_highroads.jpg    
spectral library file: data_dir/\<PATH_TO_SPECTRAL_LIBRARY_FILE\>.hdr  

## Instructions for running the program ## 
1. Run prepare_data.py to preprocess data from the Berlin-Urban-Gradient dataset and create test scenes. As the original dataset is huge, it is divided into relatively smaller sub-scenes. These test scenes are saved as '\<ds_name\>.pkl' files in data directory. Two sample datasets are provided in data directory - namely hymap02_ds02_sub_img.pkl and hymap02_ds02_sub_img1.pkl. These can be read using cPickle libray into numpy matrices. The corresonding ground truth files '\<ds_name\>_infra_GT_highroads.jpg', with the highroads marked manually in blue(0,0,254) should be present in the same directory. 

2. Run main.py after setting the configurations, input file names and paths in main.py. By default, the script calls the nearest neighbor classifier. To run the spectral unmixing classifier comment/uncomment the call to classify_nn/classify_spectra_unmix in main.py. For fresh run, the program creates a graph from the input image and saves it in data directory as edges_\<ds_name\>_\<dist_flag\>.dat. These can be reused from subsequent runs for faster execution by setting run_flag='E'. The colored segmented image, classifier output image, shape score image and the final road labelled image are generated in output directory.


 




