import numpy as np
import imutils
import cv2
from params import *

# takes RGB image and class label (assumes single label image) and generates resized image and
# binary mask with values [class_label, 18], signifying where the class is in the image and
# where the non-class pixels are in the image.
def get_resized_image_and_mask_label(image, class_label, print_report=None):
    # extract green channel
    img = image
    
    # resize and pad image to desired square size (resized if larger or smaller; padded if not square; aspect ratio maintained)
    h,w = img.shape[0],img.shape[1]
    if h > w:
        img = imutils.resize(img, height=DESIRED_SIZE)
        h,w = img.shape[0],img.shape[1]
        
        diff = DESIRED_SIZE - w
        left = diff // 2
        right = diff - left

        img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, 0)
        h,w = img.shape[0],img.shape[1]
    elif w > h:
        img = imutils.resize(img, width=DESIRED_SIZE)
        h,w = img.shape[0],img.shape[1]
        
        diff = DESIRED_SIZE - h
        top = diff // 2
        bot = diff - top
        
        img = cv2.copyMakeBorder(img, top, bot, 0, 0, cv2.BORDER_CONSTANT, 0)
        h,w = img.shape[0],img.shape[1]
    else:
        img = cv2.resize(img, (DESIRED_SIZE,DESIRED_SIZE))
        h,w = img.shape[0],img.shape[1]
    

    assert w==h, f"{w} != {h} ... original image size: {image.shape}; image width and height must be the same!"

    # calculate threshold for binarizing
    green = img[:,:,1]
    bin_thresh = (np.mean(green)) + (STDs*np.std(green))
    
    # generate class-based mask from image
    keep = green > bin_thresh
    remove = green <= bin_thresh
    mask = np.zeros(green.shape, dtype=np.uint8)
    mask[remove] = NEGATIVE
    mask[keep] = class_label
    
    #_,mask = cv2.threshold(img,bin_thresh,class_label,cv2.THRESH_BINARY)
    
    if print_report:
        print(print_report)

    return img,mask


def maskpred(preds):
    masks = np.array([np.argmax(pred, axis=-1) for pred in preds])
    return masks