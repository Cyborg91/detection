#!/usr/bin/env python
# encoding: utf-8
'''
@author: 郑祥忠
@license: (C) Copyright 2013-2019, 海格星航
@contact: dylenzheng@gmail.com
@project: Context-Aware_Crowd_Counting-pytorch
@file: my_dataset.py
@time: 6/26/19 1:37 PM
@desc:
'''
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
import random

class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,gt_dmap_root,gt_downsample=1,phase='train'):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        phase: train or test
        '''
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_downsample=gt_downsample
        self.phase=phase

        self.img_names=[filename for filename in os.listdir(img_root) if os.path.isfile(os.path.join(img_root,filename))]
        self.n_samples=len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        img=plt.imread(os.path.join(self.img_root,img_name))/255

        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        gt_dmap=np.load(os.path.join(self.gt_dmap_root,img_name.replace('.jpg','.npy')))
        
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[:,::-1]
            gt_dmap=gt_dmap[:,::-1]
        
        if self.gt_downsample>1:
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            img=img.transpose((2,0,1))
            gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
            gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample

            img_tensor=torch.tensor(img,dtype=torch.float)
            img_tensor=transforms.functional.normalize(img_tensor,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)

        return img_tensor,gt_dmap_tensor,img_name