#josh.anish1@gmail.com
#the code has been created with an intention of previewing the entire project
#contributions are welcomed!!!
import cv2
import numpy as np
import os
import sys
import pandas as pd
# import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D

from keras.layers import MaxPooling2D
from keras.layers import Dense

from keras.layers import Flatten,Dropout
from keras.models import model_from_json

img_file = sys.argv[1]
directory = sys.argv[2]



def predictor(img_file):
    img = cv2.imread(img_file)
    resize = cv2.resize(img,(64,64))
    #resize = np.expand_dims(resize,axis=0)
    

    img_fin = np.reshape(resize,[1,64,64,3])
    json_file = open('model/binaryfas10.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/binaryfashion.h5")
    # print("Loaded model from disk")
    
    prediction = loaded_model.predict_classes(img_fin)
    
    prediction = np.squeeze(prediction,axis=1)
    predict = np.squeeze(prediction,axis=0)
    return int(predict)

"""Neural Network Decoding"""
""" The coordinates are created and trained"""
"""-----------------"""
image_width = 300
image_height = 500



def path_file(file):
    return str(file)

def nn(img_file):
    predict = predictor(img_file)
    file = path_file("annotation.csv")
    reader = pd.read_csv(file)
    print(predict)
    

    img = cv2.imread(img_file)
    img = cv2.resize(img,(image_width,image_height))
    #seg = image(image,reader.x1[predict],reader.y1[predict],reader.x2[predict],reader.y2[predict],reader.i[predict])
    

    mask = np.zeros(img.shape[:2],np.uint8)   
    
    bgdModel = np.zeros((1,65),np.float64)
    
    fgdModel = np.zeros((1,65),np.float64)
    

    rect = (reader.x1[predict],reader.y1[predict],reader.x2[predict],reader.y2[predict])
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,reader.i[predict],cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    img_cut = img*mask2[:,:,np.newaxis]
    cv2.imwrite(directory+"/"+str(img_file),img_cut)

nn(img_file)
