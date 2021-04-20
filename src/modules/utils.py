import numpy as np
import cv2 as cv

from modules.data import data_info, train_info


def id_from_filename(image_file):
    return image_file.split('.')[0]


def poly_mask_to_img(data, id):
    h = data_info[data_info['image_file'] == id].iloc[0]['height_pixels']
    w = data_info[data_info['image_file'] == id].iloc[0]['width_pixels']
    mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    for mask_poly in data[id]['mask_poly']:
        mask_img = cv.polylines(mask_img, mask_poly.astype(np.int32), True, (255, 0, 0), thickness=30)
    return mask_img


def overlay_image_mask(image, mask, mask_color=(255, 0, 0), alpha=1.0):
    im_f = image.astype(np.float32)
    mask_col = np.expand_dims(np.array(mask_color) / 255.0, axis=(0, 1))
    return (im_f + alpha * mask * (np.mean(0.8 * im_f + 0.2 * 255, axis=2, keepdims=True) * mask_col - im_f)).astype(
        np.uint8)


def make_grid(shape, window=1024, min_overlap=100):
    y, x = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


def get_tile(image, mask, x, y, tile_size, scale=1.0):
    x = round(x * scale)
    y = round(y * scale)
    image_s = image[y:y + tile_size, x:x + tile_size, :]
    mask_s = mask[y:y + tile_size, x:x + tile_size, :]
    return image_s, mask_s


def rle_to_image(image_file):
    image_id = id_from_filename(image_file)
    train_df = train_info.loc[train_info['id'] == image_id]
    rle = train_df['encoding'].values[0] if len(train_info) > 0 else None

    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    image_shape = (data_info[data_info['image_file'] == image_file].iloc[0]['width_pixels'],
                   data_info[data_info['image_file'] == image_file].iloc[0]['height_pixels'])
    image = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        image[lo:hi] = 1

    return np.expand_dims(image.reshape(image_shape).T, -1)
