import numpy as np
import pandas as pd
import os
import glob
import cv2
import matplotlib.pyplot as plt
import imutils
from collections import Counter
from keras.optimizers import Adam
from keras import backend as K
import tensorflow_addons as tfa
from focal_loss import SparseCategoricalFocalLoss

from utils import get_resized_image_and_mask_label
from model import Unet
from params import *


SEG_IMAGES_DIR = './hpa-cell-tiles-sample-balanced-dataset/cells/'
segmented_cell_images = glob.glob(SEG_IMAGES_DIR+'*')
cell_df = pd.read_csv('./hpa-cell-tiles-sample-balanced-dataset/cell_df.csv')


# filter cell_df to only images with single label
single_label_cell_df = cell_df.loc[cell_df.image_labels.isin([str(x) for x in range(0,20)])]

# sample images down (remove to get full sample)
sample = single_label_cell_df.sample(SAMPLE_SIZE)

# load images and process into resized images and masks across sample
images,masks = list(zip(
    *[
        get_resized_image_and_mask_label(
            cv2.imread(SEG_IMAGES_DIR+str(p1)+'_'+str(p2)+'.jpg'),
            int(lbl),
            print_report=f"processing image {i}/{SAMPLE_SIZE}..." if i%PRINT_REPORT_SUBDIVISION == 0 else None) for
        (i,(p1,p2,lbl)) in 
        enumerate(list(zip(sample['image_id'],sample['cell_id'],sample['image_labels'])))
    ]
))

idx = round(TRAIN_PROPORTION*SAMPLE_SIZE)


valid_images,valid_labels = np.array(images[0:idx],dtype=np.uint8),np.array(masks[0:idx],dtype=np.uint8)
train_images,train_labels = np.array(images[len(valid_images):],dtype=np.uint8),np.array(masks[len(valid_labels):],dtype=np.uint8)

print(str(train_images.shape),'images and',str(train_labels.shape),'masks')

xval,yval = np.array(valid_images,dtype=np.float32),np.array(valid_labels,dtype=np.float32)
xtrain,ytrain = np.array(train_images,dtype=np.float32),np.array(train_labels,dtype=np.float32)
print(str(xtrain.shape),'images and',str(ytrain.shape),'masks')


unet = Unet(DESIRED_SIZE, DESIRED_SIZE, nclasses=NUM_CLASSES, filters=UNET_FILTERS)
print(unet.output_shape)

unet.summary()
# Setting BCE to False to use focal loss
BCE = False
print(xtrain.shape,train_images.shape)
if BCE:
    unet.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    history = unet.fit(train_images,train_labels,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(valid_images,valid_labels))
else:
    unet.compile(optimizer='adam', loss=SparseCategoricalFocalLoss(gamma=2), metrics=['accuracy'])
    history = unet.fit(xtrain,ytrain,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(xval,yval)) 

unet.save(f"unet-SAMPLESIZE{SAMPLE_SIZE}-EPOCHS{EPOCHS}.h5")

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label="Training loss")
plt.plot(epochs, val_loss,"b",label="Validation loss")
plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
