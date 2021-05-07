import numpy as np
import pandas as pd
import os
import glob
import cv2
import argparse
from focal_loss import SparseCategoricalFocalLoss
import matplotlib
import matplotlib.pyplot as plt
import copy

from model import Unet
from params import *
from utils import get_resized_image_and_mask_label, maskpred


parser = argparse.ArgumentParser(description='Generate predictions based on custom weights. HPA Cell Segmentationa and Masking Project')
parser.add_argument('weights', help='path to .h5 weights file')
parser.add_argument('images', help='path to directory containing JPG images of single cells to predict masks for')
parser.add_argument('-g', '--generate_images', help='saves representative image of mask in addition to the mask prediction', action="store_true")
args = parser.parse_args()


assert os.path.isfile(args.weights), "Could not find weights file!"
assert os.path.isdir(args.images), "Could not find images directory!"

unet = Unet(DESIRED_SIZE, DESIRED_SIZE, nclasses=NUM_CLASSES, filters=UNET_FILTERS)
unet.load_weights(args.weights)
unet.compile(optimizer='adam', loss=SparseCategoricalFocalLoss(gamma=2), metrics=['accuracy'])

image_paths = glob.glob(args.images+'/*.jpg')
images = [x[0] for x in [get_resized_image_and_mask_label(cv2.imread(p),NEGATIVE) for p in image_paths]]

mpreds = unet.predict(np.array(images, dtype=np.uint8))
preds = maskpred(mpreds)

cmap = copy.copy(plt.cm.get_cmap("gist_rainbow", 20))
cmap.set_under(color="black")

print(f"Generating predictions for {len(images)} images...")

if args.generate_images:
    for img_path,pred in zip(image_paths,preds):
        np.savez(f"{img_path.split('.jpg')[0]}_PRED.npz", pred)

        pred[pred == 18] = -1

        plt.imshow(pred, cmap=cmap, interpolation='none', vmin=0, vmax=19)
        plt.colorbar(ticks=range(0,20))
        plt.axis('off')
        plt.savefig(f"{img_path.split('.jpg')[0]}_PRED_IMG.png")
        plt.clf()

else:
    for img_path,pred in zip(image_paths,preds):
        np.savez(f"{img_path.split('.jpg')[0]}_PRED.npz", pred)

print(f"{len(image_paths)} predictions produced.")
