#!/usr/bin/env python
# encoding: utf-8
'''
@author: 郑祥忠 
@license: (C) Copyright 2013-2019, 海格星航
@contact: dylenzheng@gmail.com 
@project: Context-Aware_Crowd_Counting-pytorch
@file: demo_img.py
@time: 6/27/19 2:37 PM
@desc: test img for one img
'''
import torch
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
import time
from torchvision import transforms
from cannet import CANNet
from my_dataset import CrowdDataset

model=CANNet()
model_param_path = './checkpoints/cvpr2019CAN_353model.pth'
model.load_state_dict(torch.load(model_param_path))
model.cuda()
model.eval()
torch.backends.cudnn.enabled = False

def read_img(img_path,gt_downsample):
    img = plt.imread(img_path)/255 #rgb
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), 2)

    if gt_downsample > 1:
        ds_rows = int(img.shape[0] // gt_downsample)
        ds_cols = int(img.shape[1] // gt_downsample)
        img = cv2.resize(img, (ds_cols * gt_downsample, ds_rows * gt_downsample))
        img = img.transpose((2, 0, 1)) # numpy
        img_tensor = torch.tensor(img, dtype=torch.float)
        img_tensor = transforms.functional.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return img_tensor

def estimate_density_map(img):
    '''
    Show one estimated density-map.
    img: numpy img
    '''
    img = img.unsqueeze(0)
    img=img.cuda()
    et_dmap=model(img).detach()
    et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
    pred_frame = plt.gca()
    plt.imshow(et_dmap,cmap=CM.jet)
    #plt.show()
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    plt.savefig( 'result/'+'_out' + '.png',bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()

if __name__=="__main__":
    img_path = 'test_img/IMG_1.jpg'
    img_tensor = read_img(img_path,8)
    st = time.time()
    estimate_density_map(img_tensor)
    et = time.time()
    print('tt = ',et-st)