# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : views.py
# @time    : 19-07-11 下午16:20
# @desc    : 人头检测请求入口文件
'''
from django.shortcuts import render
from django.shortcuts import render, redirect,render_to_response
from django.http import HttpResponse
from django.http import JsonResponse

import os
import cv2
import traceback
from PIL import Image
import json
import time
import base64
import numpy as np
import threading

import server.init as init
from detect.face_detect import detect

'''
完成所有视频分析后台服务的启动
'''
print('完成所有视频分析后台服务的启动')


def argparse_base64_opencv(img_base64):
    '''
    解析base64格式的img为python's numpy格式的img
    :param img_base64: base64格式的img
    :return: python numpy格式的img
    '''
    img_b64decode = base64.b64decode(img_base64)        # base64解码
    img_array = np.fromstring(img_b64decode,np.uint8)   # 转换np序列
    img_numpy=cv2.imdecode(img_array,cv2.COLOR_BGR2RGB) # 转换Opencv格式
    return img_numpy


def face_detect(request):
    '''
    功能:人脸检测框和人脸数量接口
    服务器Http协议web网络被调用接口
    '''
    if request.method == "POST":
        try:
            request_dict = json.loads(request.body.decode())
        except:
            request_dict = request.POST
        img_base64 = request_dict.get('img_base64')
        img = argparse_base64_opencv(img_base64)
        call_json = detect(img)
        return JsonResponse(call_json)
