import json
import os
import glob

import numpy as np
import pandas as pd

from modules.consts import TRAIN_HOME, HOME

anatomical_files = [os.path.basename(f) for f in glob.glob(TRAIN_HOME + "*.json") if "anatomical" in f]
masks_files = [os.path.basename(f) for f in glob.glob(TRAIN_HOME + "*.json") if "anatomical" not in f]
image_files = [os.path.basename(f) for f in glob.glob(TRAIN_HOME + "*.tiff")]
data_info = pd.read_csv(os.path.join(HOME, "HuBMAP-20-dataset_information.csv"))
train_info = pd.read_csv(os.path.join(HOME, "train.csv"))


def load_mask_poly(data):
    for id in image_files:
        with open(TRAIN_HOME + data_info[data_info['image_file'] == id].iloc[0]['glomerulus_segmentation_file']) as jsonfile:
            polys = json.load(jsonfile)
            data[id]['mask_poly'] = []
            for idx in range(len(polys)):
                if polys[idx]['properties']['classification']['name'] == 'glomerulus':
                    geom = np.array(polys[idx]['geometry']['coordinates'])
                    data[id]['mask_poly'].append(geom)
    return data


def load_data():
    data = {img: {} for img in image_files}
    return load_mask_poly(data)
