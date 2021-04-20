import cv2 as cv

from modules.data import image_files
from modules.tiff import open_tiff_file
from modules.utils import rle_to_image, make_grid, id_from_filename, get_tile, overlay_image_mask


def generate_all_samples():
    for img_file in image_files:
        generate_samples(img_file)


def generate_samples(img_file, window=1024, scale=1.0):
    mask = rle_to_image(img_file)
    image = open_tiff_file(img_file)
    size = image.shape
    boxes = make_grid((size[0], size[1]), window=window)
    file_id = id_from_filename(img_file)
    for i, box in enumerate(boxes):
        x, y = box[0], box[2]
        image_s, mask_s = get_tile(image, mask, x, y, window, scale)
        o = overlay_image_mask(image_s, mask_s)
        mask_s = mask_s * 255
        cv.imwrite(f'data/overlays/{file_id}_{i:03}.png', cv.cvtColor(o, cv.COLOR_RGB2BGR))
        cv.imwrite(f'data/masks/{file_id}_{i:03}.png', cv.cvtColor(mask_s, cv.COLOR_GRAY2BGR))
        cv.imwrite(f'data/images/{file_id}_{i:03}.png', cv.cvtColor(image_s, cv.COLOR_RGB2BGR))
        del image_s, mask_s, o
    del image
