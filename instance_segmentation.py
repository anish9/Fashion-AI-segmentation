import cv2
import numpy as np
import os
import sys

img_file = sys.argv[1]
save_dir = sys.argv[2]

img_size_x,img_size_y = 300,500
op_img_x,op_img_y = 1600,2300

contrast   = 120

brightness = 0

pixel_factor = 127

default_tree = 40

create = cv2.GC_INIT_WITH_RECT


def segment(img,x,y,w,z,iteration,savefile,rgbf,rgbb):
    imgs = cv2.imread(img)
    img = cv2.resize(imgs,(img_size_x,img_size_y))
    
    mask = np.zeros(img.shape[:2],np.uint8)   
    bgdModel = np.zeros((1,65),np.float64)
    
    fgdModel = np.zeros((1,65),np.float64)
    coord = (x,y,w,z)
    
    cv2.grabCut(img,mask,coord,bgdModel,fgdModel,iteration,create)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_cut = img*mask2[:,:,np.newaxis]
    rese = cv2.resize(img_cut,(op_img_x,op_img_y))
    img_cont = np.int16(rese) 

    img_cont = img_cont*(contrast/1) - contrast + brightness
    img_cont = np.clip(img_cont, 0, 255)
    opr = np.uint8(img_cont)
    
    op1 = opr[np.where((opr == [255,255,255]).all(axis = 2))] = rgbf
    op2 = opr[np.where((opr == [0,0,0]).all(axis = 2))] = rgbb
    
    cv2.imwrite(save_dir+str(savefile),opr)

p1 = 80-5
p2 = 10-5
p3 = 195-5
p4 = 420+80
x_po,y_po,w_po,z_po = p1,p2,p3,p4

segment(img_file,x_po,y_po,w_po,z_po,default_tree,img_file,[255,0,0],[0,255,0])
