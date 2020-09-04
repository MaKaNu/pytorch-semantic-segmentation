""" This Module creates Dataset based on the BoxSupDataset Package
"""
from __future__ import absolute_import
from BoxSupDataset.nasa_box_sup_dataset import NasaBoxSupDataset
from BoxSupDataset.transforms.utils import ToTensor
from torchvision import transforms
from PIL import Image
import numpy as np

from . import dataset_errors

NUM_CLASSES = 7
IGNORE_LABEL = 255
ROOT = 'D:/Mitarbeiter/Kaupenjohann/09_GIT/PyTorch_Nasa_Dataset/data/TestBatch'

# Try to load the Dataset
try:
    Dataset = NasaBoxSupDataset(
    classfile='classes_bxsp.txt',
    rootDir=ROOT,
    transform=transforms.Compose(
        [ToTensor(),
        ]
    ))
except AssertionError as err:
    raise dataset_errors.LoadingError(err.args[0].split(' ',1)[0]) from err

# color map
# 0=background, 1=sand, 2=soil, 3=bedrock, 4=big rocks, 5=sky, 6=robot

palette = [0, 0, 0]

for obj_cls in Dataset.classes.iterrows():
    palette.extend(list(obj_cls[1])[1:])
print(palette)

ZERO_PAD = 256 * 3 - len(palette)
for i in range(ZERO_PAD):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask
