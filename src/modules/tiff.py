import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from osgeo import gdal


def open_tiff_file(path):
    dataset = gdal.Open(path)
    print(str(dataset.RasterCount) + ' dimensions')
    bands = []
    for band in range(dataset.RasterCount):
        bands.append(dataset.GetRasterBand(band + 1).ReadAsArray())

    return np.dstack(tuple(bands))


def show_img_from_tiff(path):
    img = open_tiff_file(path)
    img_resized = cv.resize(img, None, fx=0.5, fy=0.5)
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(img_resized)
