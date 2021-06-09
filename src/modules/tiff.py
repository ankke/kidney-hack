import numpy as np
import matplotlib.pyplot as plt
import rasterio

from modules.consts import TRAIN_HOME, TEST_HOME
from modules.run_statistics import execution_time


@execution_time
def open_tiff_file(image_file, base_path=TRAIN_HOME):
    with rasterio.open(base_path + image_file) as file:
        if file.count == 3:
            image = file.read([1, 2, 3]).transpose(1, 2, 0).copy()
        else:
            h, w = (file.height, file.width)
            subdatasets = file.subdatasets
            if len(subdatasets) > 0:
                image = np.zeros((h, w, len(subdatasets)), dtype=np.uint8)
                for i, subdataset in enumerate(subdatasets, 0):
                    with rasterio.open(subdataset) as layer:
                        image[:, :, i] = layer.read(1)
    return image


@execution_time
def show_img_from_tiff(image_file):
    img = open_tiff_file(image_file)
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.imshow(img)
