# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : face_detect.py
# @time    : 19-07-12 20:21
# @desc    : 检测帧中的所有人脸用以识别
'''
import ctypes
import cv2
from threading import Lock
import time

# 加载人脸检测so库
ll = ctypes.cdll.LoadLibrary
lib = ll("detect/libfacedetection.so")

# 人脸检测锁
mutex=Lock()

class StructPointer(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int),("boxes", ctypes.c_int *1000)]
lib.facedetect_cnn_y.restype = ctypes.POINTER(StructPointer)


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


def detect(img):
    '''
    功能：人脸检测
    img: 图片数据
    return: 人脸检测框和人脸数量
    '''
    try:

        mutex.acquire()

        print('start*********************sleep 10 S')
        time.sleep(10)

        print('end*********************sleep 10 S')

        rows, cols, channel = img.shape
        dataptr = img.ctypes.data_as(ctypes.c_char_p) # 转换成c++格式类型

        # 人脸检测结果
        print("人脸检测进行中......")
        face_detection = lib.facedetect_cnn_y(dataptr, rows, cols, channel)
        boxes = face_detection.contents.boxes
        num = face_detection.contents.num

        if num == 0:
            faceRect = []
            call_jeson(num,faceRect)

        else:
            ret_data = []
            xmin,ymin,xmax,ymax=0,0,0,0
            for i in range(num):
                xmin = boxes[i*5+0]
                ymin = boxes[i*5+1]
                xmax = xmin + boxes[i*5+2]
                ymax = ymin + boxes[i*5+3]

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
        print('odd error')
        cv2.imwrite('result_img/error_img.jpg', img)
        return {"code":1,
                "msg":"system error",
                "data":{"faceRect":[]}}

