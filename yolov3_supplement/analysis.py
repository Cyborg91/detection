#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#this code is to extract the yolov3 train log
'''
import inspect
import os
import random
import sys
'''
def extract_log(log_file,new_log_file,key_word):
    f=open(log_file,'r')
    train_log=open(new_log_file,'w')
    for line in f:
        if 'Syncing' in line:        #多gpu同步信息，我就一个GPU,这里是可以不要的。
            continue
        if 'nan' in line:            #包含nan的不要
            continue
        if key_word in line:         #包含关键字
            train_log.write(line)
    f.close()
    train_log.close()
    
extract_log('train_yolov3.log','DJI_yolov3_train_loss.txt','images')
extract_log('train_yolov3.log','DJI_yolov3_train_iou.txt','IOU')
