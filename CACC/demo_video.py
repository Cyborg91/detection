#!/usr/bin/env python
# encoding: utf-8
'''
@author: 郑祥忠 
@license: (C) Copyright 2013-2019, 海格星航
@contact: dylenzheng@gmail.com 
@project: Context-Aware_Crowd_Counting-pytorch
@file: demo_video.py
@time: 6/27/19 2:37 PM
@desc: test videos for serials frame
'''
import torch
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from torchvision import transforms
from cannet import CANNet
from my_dataset import CrowdDataset

model=CANNet()
model_param_path = './checkpoints/cvpr2019CAN_353model.pth'
model.load_state_dict(torch.load(model_param_path))
model.cuda()
model.eval()
torch.backends.cudnn.enabled = False

def read_img(img,gt_downsample):
    img = img / 255
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), 2)

    if gt_downsample > 1:
        ds_rows = int(img.shape[0] // gt_downsample)
        ds_cols = int(img.shape[1] // gt_downsample)
        img = cv2.resize(img, (ds_cols * gt_downsample, ds_rows * gt_downsample))
        img = img.transpose((2, 0, 1))
        img_tensor = torch.tensor(img, dtype=torch.float)
        img_tensor = transforms.functional.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return img_tensor

def estimate_density_map(img):
    '''
    Show one estimated density-map.
    img: numpy_img
    '''
    img = img.unsqueeze(0)
    img=img.cuda()
    et_dmap=model(img).detach()
    et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
    plt.ion()
    plt.imshow(et_dmap,cmap=CM.jet)
    plt.axis('off')
    plt.show()
    plt.pause(0.001)

def main():
    video_path = 'test_img/02_short.mp4'
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("couldn't open the video correctly!!!")
    while True:
        flag,frame = vid.read()
        if not flag:
            break
        img_tensor = read_img(frame,8)
        estimate_density_map(img_tensor)
main()
