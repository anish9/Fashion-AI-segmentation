# Logs Updated : 2024
# Type : Basic demo
# Node : Unknown

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

h,w = 512,512

base_model = load_model("froze.h5")

def remove_pads(im,padded_im,base_value=576):
    hw_ = list(im.shape[:2])
    h_or_w = hw_.index(max(hw_))
    if h_or_w == 0 :
        ratio  = base_value/max(hw_)
        height = base_value
        width  = int(ratio*im.shape[1])
        diff_  = (np.abs(base_value-width))//2
        crop_file = padded_im[:,diff_:base_value-diff_]
    else:
        ratio  = base_value/max(hw_)
        height = int(ratio*im.shape[0]) 
        width  = base_value
        diff_  = (np.abs(base_value-height))//2
        crop_file = padded_im[diff_:base_value-diff_,:]

    return crop_file


def predict_and_visualize(file):
    im = tf.io.read_file(file)
    im = tf.io.decode_png(im,channels=3)
    array_image = im.numpy()
    orig_h,orig_w,_ = im.shape
    im_p = tf.image.resize_with_pad(im,h,w)
    raw = np.array(im.numpy(),dtype=np.uint8)
    imb_p = np.expand_dims(im_p,axis=0)/255.
    pred = base_model.predict(imb_p,verbose=0)[0]
    unpadded_pred = remove_pads(im,pred,base_value=w) #get your mask here
    ##### visualize code #####
    # mask = np.array(unpadded_pred*255.,np.uint8)
    # mask = cv2.resize(mask,(orig_w,orig_h))[:,:,np.newaxis]
    # mask = np.repeat(mask,3,axis=-1)
    # dst = cv2.addWeighted(array_image, 0.2, mask, 0.8, 0.0)
    # cv2.imwrite("out.jpg",dst)
    return unpadded_pred


if __name__=="__main__":
    predict_and_visualize("paris_fashion_week.png")
