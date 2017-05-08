#!/usr/bin/python
#######################################################################
#   File name: convert_shapefile_pavement_pixel.py
#   Author: Soubarna Banik
#   Description: script for rasterization/ mapping shapefile polygon to pixels
#######################################################################

from osgeo import ogr
from osgeo import gdal
import numpy as np

file_name = 'data/Hyperspectral data/Berlin Urban Gradient 2009 02 additional data/02_additional_data/land_cover/LandCov_Vec_Berlin_Urban_Gradient_2009_Pavement.shp'

raster_fn = 'data/Hyperspectral data/Berlin Urban Gradient 2009 02 additional data/02_additional_data/land_cover/LandCov_Vec_Berlin_Urban_Gradient_2009_pavement.tif'

pixel_size = 3
NoData_value = 255
maskvalue = 1

driver = ogr.GetDriverByName('ESRI Shapefile')
ds = driver.Open(file_name, 0) # 0 means read-only. 1 means writeable.

layer = ds.GetLayer()
print "Feature count:", layer.GetFeatureCount()

x_min, x_max, y_min, y_max = layer.GetExtent()

print x_min, x_max
print y_min, y_max

iwidth=732
iheight=2722
xdist = x_max - x_min
ydist = y_max - y_min
xratio = iwidth/xdist
yratio = iheight/ydist

# Create the destination data source
x_res = int((x_max - x_min) / pixel_size)
y_res = int((y_max - y_min) / pixel_size)
target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, iwidth, iheight, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
band = target_ds.GetRasterBand(1)
band.SetNoDataValue(NoData_value)


# Rasterize
err = gdal.RasterizeLayer(target_ds, [maskvalue], layer, burn_values=[0], options=["ATTRIBUTE=Level_2"])
target_ds.FlushCache()


# Loop through raster
mask_arr=target_ds.GetRasterBand(1).ReadAsArray()
print "Raster array shape:", mask_arr.shape
print np.unique(mask_arr)

