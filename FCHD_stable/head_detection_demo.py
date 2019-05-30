from __future__ import division

import os
import torch as t
from src.config import opt
from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer
from PIL import Image
import numpy as np
from data.dataset import preprocess
import matplotlib.pyplot as plt 
import src.array_tool as at
from src.vis_tool import visdom_bbox
import argparse
import src.utils as utils
from src.config import opt
import cv2

THRESH = 0.01
model_path = 'checkpoints/head_detector_final'
head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
trainer = Head_Detector_Trainer(head_detector).cuda()
trainer.load(model_path)


def read_img(f):
    f = f.resize((416,234),Image.ANTIALIAS)
    f.convert('RGB')
    img_raw = np.asarray(f, dtype=np.uint8)
    img_raw_final = img_raw.copy()
    img = np.asarray(f, dtype=np.float32)

    img = img.transpose((2,0,1))
    _, H, W = img.shape
    img = preprocess(img)
    _, o_H, o_W = img.shape

    scale = o_H / H
    return img, img_raw_final, scale

def detect(img, img_raw, scale):
    img = at.totensor(img)
    img = img[None, : ,: ,:]
    img = img.cuda().float()
    pred_bboxes_, _ = head_detector.predict(img, scale, mode='evaluate', thresh=THRESH)
    for i in range(pred_bboxes_.shape[0]):
        ymin, xmin, ymax, xmax = pred_bboxes_[i,:]
        # print("xmin/scale,ymin/scale,xmax/scale,ymax/scale = ",xmin/scale,ymin/scale,xmax/scale,ymax/scale)
    #     utils.draw_bounding_box_on_image_array(img_raw,ymin/scale, xmin/scale, ymax/scale, xmax/scale)
    # plt.axis('off')
    # plt.imshow(img_raw)
    # plt.show()

video_path = 'image_test/crowd.mp4'
def main():
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOErro("couldn't open the video")
    while True:
        return_value,frame = vid.read()
        if not return_value:
            break

        st = cv2.getTickCount()
        image = Image.fromarray(frame)
        img,img_raw,scale = read_img(image)
        detect(img,img_raw,scale)
        et = cv2.getTickCount()
        tt = (et - st) / cv2.getTickFrequency()
        print('time_all = ',1000*tt)
        print('fps = ', 1//tt)
main()