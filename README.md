# RoadDetection
road detection using hyperspectral data

# Configurations Settings:
# main.py =>  
run_flag = 'N'  # for existing run : E, for new run: N  
dist_flag = 'SAD'  # SAD: SAD distance metric, EU: euclidean distance metric  
ds_name = "<DATASET_NAME>" # dataset name  
output_dir = "<PATH_TO_OUTPUT_DIRECTORY>"  
k = 100 #threshold parameters  
min_size = 10 #threshold parameters  
min_size2 = 30 #threshold parameters  


# Directory structure/Input files
data : <PATH_TO_PREPROCESSED_SUBSET_DATA>  
input spectral file: <ds_name>.pkl  
input ground truth file: <ds_name>_infra_GT_highroads.jpg  
spectral library file: <PATH_TO_SPECTRAL_LIBRARY_FILE>  

# Instructions for running the program  
1. Run prepare_data.py to preprocess data and create test scenes. Two sample datasets are provided in data directory - namely hymap02_ds02_sub_img.pkl and hymap02_ds02_sub_img1.pkl. The corresonding ground truth files should be present in the same directory.
2. Run main.py after setting the configurations in main.py. By default, the script calls the nearest neighbor classifier. To run the spectral unmixing classifier comment/uncomment the call to classify_nn/classify_spectra_unmix in main.py. 

# Dependency
python: 2.6+  
Some of the important libraries used:  
numpy  
spectral #pip install spectral (http://www.spectralpython.net/installation.html)  
skimage # pip install scikit-image (http://scikit-image.org/docs/dev/install.html)  
cPickle # pip install cPickle (https://pypi.python.org/pypi/cpickle/0.1)  
ogr, gdal # convert_shapefile_pavement_pixel: for converting shapefile polygon to pixel   
 




