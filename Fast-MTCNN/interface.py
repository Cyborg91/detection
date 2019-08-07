#!/usr/bin/env python
# encoding: utf-8
'''
@author: 郑祥忠 
@license: (C) Copyright 2013-2019, 海格星航
@contact: dylenzheng@gmail.com 
@project: MTCNN
@file: interface.py
@time: 8/7/19 7:39 PM
@desc:
'''

import ctypes
import cv2
import time
from threading import Lock

def call_jeson(num,faceRect):
    '''
    :param num: 人脸数量
    :param headRect: 人脸检测框坐标
    :return: 以人脸和人脸检测框组合成的字典信息
    '''
    call_json = {
        "code": 0,
        "msg": "success",
        "data": {
            "count": num,
            "faceRect": faceRect}
    }
    return call_json;

'''
调用程序
ll = ctypes.cdll.LoadLibrary
lib = ll('./libmtcnndetection.so')

img = cv2.imread('images/01.png')
dataptr = img.ctypes.data_as(ctypes.c_char_p)
rows, cols, channel = img.shape

class StructPointer(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int),("location", ctypes.c_int *1000)]
lib.detection.restype = ctypes.POINTER(StructPointer)
face_detection = lib.detection(dataptr,rows,cols,channel)
boxes = face_detection.contents.location
num = face_detection.contents.num

for i in range(num):
    xmin = boxes[i*4+0]
    ymin = boxes[i*4+1]
    xmax = boxes[i*4+2]
    ymax = boxes[i*4+3]
    print('xmin,ymin,xmax,ymax=',xmin,ymin,xmax,ymax)
'''
# 调用接口
def detect(img):
    '''
    功能：人脸检测
    img: 图片数据
    return: 人脸检测框和人脸数量
    '''
    try:

        mutex.acquire()
        rows, cols, channel = img.shape
        dataptr = img.ctypes.data_as(ctypes.c_char_p) # 转换成c++格式类型

        # 人脸检测结果
        print("人脸检测进行中......")
        face_detection = lib.facedetect_cnn_y(dataptr, rows, cols, channel)
        boxes = face_detection.contents.boxes
        num = face_detection.contents.num
        ret_data = []

        if num == 0:
            faceRect = []
            call_jeson(num,faceRect)


        else:
            xmin,ymin,xmax,ymax=0,0,0,0
            for i in range(num):
                xmin = boxes[i*4+0]
                ymin = boxes[i*4+1]
                xmax = boxes[i*4+1]
                ymax = boxes[i*4+1]
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0

                # 画矩形框
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)

                arr_num = [xmin,ymin,xmax,ymax]
                ret_data.append(arr_num)

        # 打印人脸数量
        print('num=',num)

        return_json = call_jeson(num,ret_data)
        mutex.release()

        return return_json

    except Exception as e:
        import traceback
        print('odd error')
        err_information = traceback.format_exc()
        traceback.print_exc()
        cv2.imwrite('result_img/error_img.jpg', img)
        return {"code":1,
                "msg":"system error",
                "data":{"faceRect":[]}}