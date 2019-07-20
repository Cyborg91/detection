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

import threading
import torch
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

import server.init as init
from detect.head_detect import detect

'''
完成所有视频分析后台服务的启动
'''
init.init_head_detect()

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

def argparse_base64_PIL(img_base64):
    '''
    解析base64格式的img为PIL.Image格式的img
    :param img_base64: base64格式的img
    :return: PIL.Image格式的img
    '''
    img_b64decode = base64.b64decode(img_b64encode)  # base64解码
    image = io.BytesIO(img_b64decode)                # 转换序列
    img_PIL = Image.open(image)
    return img_PIL

def head_detect(request):
    '''
    功能:人头检测框和人头数量接口
    服务器TCP协议web网络被调用接口
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

def head_detect_rough(request):
    '''
    功能：行人检测热点图显示接口
    服务器TCP协议web网络被调用接口
    :param request:
    :return:
    '''
    if request.method == "POST":
        try:
            request_dict = json.loads(request.body.decode())
        except:
            request_dict = request.POST

        img_base64 = request_dict.get('img_base64')
        img = argparse_base64_PIL(img_base64)
        call_json = detect(img)
        return JsonResponse(call_json)