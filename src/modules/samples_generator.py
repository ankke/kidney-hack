import cv2 as cv
import numpy as np

from modules.consts import TEST_HOME
from modules.data import image_files, image_test_files
from modules.run_statistics import execution_time
from modules.tiff import open_tiff_file
from modules.utils import rle_to_image, make_grid, id_from_filename, get_tile, overlay_image_mask, get_tile_img


@execution_time
def generate_all_samples(path, window=1024, scale=1, threshold=100, filter=False):
    for img_file in image_files:
        generate_samples(img_file, path, window, scale, threshold, filter)


@execution_time
def generate_samples(img_file, path, window=1024, scale=1, threshold=100, filter=False):
    mask = rle_to_image(img_file)
    image = open_tiff_file(img_file)
    size = image.shape
    boxes = make_grid((size[0], size[1]), window=window)
    file_id = id_from_filename(img_file)
    saved = 0
    for i, box in enumerate(boxes):
        x, y = box[0], box[2]
        image_s, mask_s = get_tile(image, mask, x, y, window)
        m = mask_s.reshape(-1).astype('int64')
        if filter and np.bincount(m, minlength=2)[1] < threshold:
            continue
        saved += 1
        o = overlay_image_mask(image_s, mask_s)
        mask_s = mask_s * 255
        o = cv.resize(o, (o.shape[1] // scale, o.shape[0] // scale))
        mask_s = cv.resize(mask_s, (mask_s.shape[1] // scale, mask_s.shape[0] // scale))
        image_s = cv.resize(image_s, (image_s.shape[1] // scale, image_s.shape[0] // scale))
        cv.imwrite(f'{path}/overlays/{file_id}_{saved:03}.png', cv.cvtColor(o, cv.COLOR_RGB2BGR))
        cv.imwrite(f'{path}/masks/{file_id}_{saved:03}.png', cv.cvtColor(mask_s, cv.COLOR_GRAY2BGR))
        cv.imwrite(f'{path}/images/{file_id}_{saved:03}.png', cv.cvtColor(image_s, cv.COLOR_RGB2BGR))
        del image_s, mask_s, o
    print(f'saved {saved}')
    del image


@execution_time
def generate_all_test_samples(path, window=1024, scale=1):
    for img_file in image_test_files:
        generate_test_samples(img_file, path, window, scale)


@execution_time
def generate_test_samples(img_file, path, window=1024, scale=1):
    file_id = id_from_filename(img_file)
    image = open_tiff_file(img_file, TEST_HOME)
    size = image.shape
    boxes = make_grid((size[0], size[1]), window=window)
    saved = 0
    for i, box in enumerate(boxes):
        x, y = box[0], box[2]
        image_s = get_tile_img(image, x, y, window)
        image_s = cv.resize(image_s, (image_s.shape[1] // scale, image_s.shape[0] // scale))
        cv.imwrite(f'{path}/images/{file_id}_{saved:03}.png', cv.cvtColor(image_s, cv.COLOR_RGB2BGR))
        saved += 1
        del image_s
    print(f'saved {saved}')
    del image
