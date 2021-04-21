import cv2 as cv
import matplotlib.pyplot as plt

from modules.data import image_files
from modules.run_statistics import execution_time
from modules.tiff import open_tiff_file
from modules.utils import rle_to_image, make_grid


@execution_time
def plot(img_file, with_overlay=True, with_grid=False, window=1024, scale=20, cmap='gray', alpha=.25, min_overlap=100):
    image = open_tiff_file(img_file)
    print('image loaded')
    mask = rle_to_image(img_file)
    print('mask decoded')
    size = image.shape
    if with_grid:
        boxes = make_grid((size[0], size[1]), window=window, min_overlap=min_overlap)
        for i, box in enumerate(boxes):
            x1, y1 = box[0], box[2]
            x2, y2 = box[1], box[3]
            image = cv.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=8)
        print('grid generated')
    image = cv.resize(image, (image.shape[1] // scale, image.shape[0] // scale))
    mask = cv.resize(mask, (mask.shape[1] // scale, mask.shape[0] // scale))
    print('resized')
    plt.subplots(figsize=(42, 40))
    plt.imshow(image)
    if with_overlay:
        plt.imshow(mask, alpha=alpha, cmap=cmap)
    plt.grid(None)
    plt.axis('off')
    plt.show()
    del image


def plot_all(with_overlay=True, with_grid=False, window=1024, scale=20, cmap='gray', alpha=.25, min_overlap=100):
    for img_file in image_files:
        print(f'ploting {img_file}')
        plot(img_file, with_overlay=with_overlay, with_grid=with_grid, scale=scale, window=window, cmap=cmap,
             alpha=alpha, min_overlap=min_overlap)
