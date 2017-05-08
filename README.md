# RoadDetection
road detection using hyperspectral data

# Directory structure
data = <PATH_TO_PREPROCESSED_SUBSET_DATA>
data/Hyperspectral data = <PATH_TO_ORIGINAL_SPECTRAL_DATASET>

# Configurations Settings:
# main.py =>
run_flag = 'N'  # for existing run : E, for new run: N
dist_flag = 'SAD'  # SAD: SAD distance metric, EU: euclidean distance metric
ds_name = "<DATASET_NAME>.pkl" # dataset name
output_dir = "PATH_TO_OUTPUT_DIRECTORY" # output directory
k = 100 #threshold parameters
min_size = 10 #threshold parameters
min_size2 = 30 #threshold parameters

# Instructions for running the program
Run main.py after setting the configurations in main.py. By default, the script calls the nearest neighbor classifier. To run the spectral unmixing classifier comment/uncomment the call to classify_nn/classify_spectra_unmix in main.py. 

# Dependency
python: 2.6+
Some of the important libraries used:
numpy
spectral #pip install spectral (http://www.spectralpython.net/installation.html)
skimage # pip install scikit-image (http://scikit-image.org/docs/dev/install.html)

# convert_shapefile_pavement_pixel: for converting shapefile polygon to pixel 
ogr 
gdal




