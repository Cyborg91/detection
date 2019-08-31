# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : head_detect.py
# @time    : 19-07-12 20:21
# @desc    : 检测帧中的所有人头用以计数
'''
import os
import cv2
import numpy as np
import time
import requests
from urllib.request import urlretrieve
from threading import Lock

import detect.denoise.pbcvt as denoise
import detect.enhance.pbcvt as enhance
from detect.utils import detection

'''
THRESHOLD
低光增强和去噪处理超参
花城汇球机轮巡获得超参
更换场景后需要重新获取
'''
THRESHOLD = 120
mutex = Lock()


def resize_wrapper(ratio=0.8):
    """
    a resize wrapper, resize the input image, and return the correct location
    add by shengyang at 2019.08.30 13:59
    :param ratio: resize rate
    :return:
    """
    def resize(func):
        def function(*args, **kwargs):
            """
            注意，装饰的方法入第一个参数为输入opencv打开的图片，而且不能以关键字参数调用
            若需要支持关键字参数调用，需要相应的此处
            :param args:
            :param kwargs:
            :return:
            """
            img = args[0]
            # print("in wrapper, ori size={}".format(img.shape))
            new_img = cv2.resize(src=img, dsize=(round(img.shape[1]*ratio),
                                                 round(img.shape[0]*ratio)),
                                 interpolation=cv2.INTER_CUBIC)
            # print("in wrapper, new size={}".format(new_img.shape))
            ret = func(new_img)
            # print("in wrapper, ori rect={}".format(ret['data']['headRect']))
            ret['data']['headRect'] = [[round(ratio*i) for i in rect]
                                       for rect in ret['data']['headRect']]
            # print("in wrapper, new rect={}".format(ret['data']['headRect']))
            return ret
        return function
    return resize


def call_jeson(num,headRect):
    '''
    :param num: 人头数量
    :param headRect: 人头检测框坐标
    :return: 人头数量和人头检测框组合成的字典信息
    '''
    call_json = {
        "code": 0,
        "msg": "success",
        "data": {
            "count": num,
            "headRect": headRect}
    }
    return call_json;

@resize_wrapper(0.8)
def detect(img):
    '''
    功能：人头检测
    img: 图片数据
    return: 人头检测框和人头数量
    '''
    try:
        mutex.acquire()	
        # 将BGR值转换为灰度值
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        pix_threshold= gray.sum()/gray.size
        print('pix_threshold=',pix_threshold)

        #光线变暗时做低光增强和去噪处理
        if pix_threshold < THRESHOLD:
            img = enhance.enhance_msRestinex(img)
            img = denoise.curvature_filter_sse(img, 5)

        # 人头检测结果
        boxes = detection(img)
        print("人头计数进行中......")

        ret_data = []
        num = 0
        xmin,ymin,xmax,ymax=0,0,0,0
        for i in range(boxes.shape[0]):
            xmin = int(boxes[i][0])
            ymin = int(boxes[i][1])
            xmax = int(boxes[i][2])
            ymax = int(boxes[i][3])
            score = boxes[i][4]
            if score < 0.15:
                continue
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            num = num + 1

            # 画矩形框
            # cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)

            arr_num = [xmin,ymin,xmax,ymax]
            ret_data.append(arr_num)


        # 写入图片
        # cv2.imwrite('result_img/'+'{}'.format(num)+'.jpg',img)

        # 打印人头数量
        print('num=',num)

        return_json = call_jeson(num,ret_data)
        mutex.release()
        return return_json

    except Exception as e:
        import traceback
        print('detection errors')
        err_information = traceback.format_exc()
        traceback.print_exc()
        return {"code":1,
                "msg":"system error",
                "data":{"headRect":[],
                        "err_information":"detection unknow error"}}
