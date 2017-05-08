# RoadDetection
road detection using hyperspectral data

# Directory structure
data = <PATH_TO_PREPROCESSED_SUBSET_DATA>

# Configurations settings:
main.py =>
run_flag = 'N'  # for existing run : E, for new run: N
dist_flag = 'SAD'  # SAD: SAD distance metric, EU: euclidean distance metric
ds_name = "<DATASET_NAME>.pkl" # dataset name
output_dir = "PATH_TO_OUTPUT_DIRECTORY" # output directory

# Threshold parameters
k = 100
min_size = 10
min_size2 = 30





