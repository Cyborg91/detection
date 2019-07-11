import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio

from PIL import Image, ImageDraw # call PIL  lib
from pyramid import build_sfd
from layers import *
import cv2
import time
import numpy as np
import math
import argparse
import datetime

def parser_fun():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
    # 行人数量在150~220之间使用阈值0.2,高于350使用0.15
    parser.add_argument('--probability', default='0.15')  # 0.15, 0.2, 0.25，0.3
    parser.add_argument('--resume', default='weights/Umbrella_Res50_pyramid.pth')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.set_device(0)

def net_build():
    ssd_net = build_sfd('test', 640, 2)
    net = ssd_net
    net.load_state_dict(torch.load(args.resume)) # load model parameters only
    net.cuda()
    net.eval() # set model as evaluations only dropout and BatchNorm exsist
    print('Finished loading model!')


# method 1
def detect_face(image, shrink):
    x = image
    if shrink != 1:
        x = cv2.resize(image, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR) #双线性插值算法
    width = x.shape[1]
    height = x.shape[0]
    x = x.astype(np.float32)
    x -= np.array([104, 117, 123], dtype=np.float32)

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    # x = Variable(x.cuda(), volatile=True)
    with torch.no_grad():
        x = Variable(x.cuda())


    net.priorbox = PriorBoxLayer(width, height) # predict coordinations
    y = net(x) # cost time too much #
    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    boxes = [] # coordinations
    scores = [] # confidence
    for i in range(detections.size(1)): # detection_size(1) = 2
        j = 0
        while detections[0, i, j, 0] >= 0.01:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            boxes.append([pt[0], pt[1], pt[2], pt[3]])
            scores.append(score)
            j += 1
            if j >= detections.size(2):
                break

    det_conf = np.array(scores)
    boxes = np.array(boxes)

    if boxes.shape[0] == 0:
        return np.array([[0, 0, 0, 0, 0.001]])

    det_xmin = boxes[:, 0] / shrink
    det_ymin = boxes[:, 1] / shrink
    det_xmax = boxes[:, 2] / shrink
    det_ymax = boxes[:, 3] / shrink
    det = np.column_stack((det_xmin, det_ymin, det_xmax, det_ymax, det_conf))

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det


# method 2
def multi_scale_test(image, max_im_shrink):
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = detect_face(image, st)
    if max_im_shrink > 0.75:
        det_s = np.row_stack((det_s, detect_face(image, 0.75)))
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]

    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    # print('bt = ',bt)
    det_b = detect_face(image, bt)

    # enlarge small iamge x times for small face
    if max_im_shrink > 1.5:
        det_b = np.row_stack((det_b, detect_face(image, 1.5)))

    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:  # and bt <= 2:
            det_b = np.row_stack((det_b, detect_face(image, bt)))
            bt *= 2

        det_b = np.row_stack((det_b, detect_face(image, max_im_shrink)))

    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]

    return det_s, det_b


# method 3
def multi_scale_test_pyramid(image, max_shrink):
    # shrink detecting and shrink only detect big face
    det_b = detect_face(image, 0.25)
    index = np.where(
        np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1)
        > 30)[0]
    det_b = det_b[index, :]

    st = [1.25, 1.75, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_shrink):
            det_temp = detect_face(image, st[i])
            # enlarge only detect small face
            if st[i] > 1:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) < 100)[0]
                det_temp = det_temp[index, :]
            else:
                index = np.where(
                    np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1,
                               det_temp[:, 3] - det_temp[:, 1] + 1) > 30)[0]
                det_temp = det_temp[index, :]
            det_b = np.row_stack((det_b, det_temp))
    return det_b


# method4
def flip_test(image, shrink):
    image_f = cv2.flip(image, 1)
    det_f = detect_face(image_f, shrink)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = image.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = image.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    return det_t


def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU calculation
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        merge_index = np.where(o >= 0.7)[0]  # IOU > =0.7 merge_index.shape=(1,) merge_index = 0

        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        # print('100')

        # if merge_index.shape[0] <= 1: # question merge_index = 0 loop
        #     continue

        # print('100000')

        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score


        try:
            dets = np.row_stack((dets, det_accu_sum))

        except:
            dets = det_accu_sum

    dets = dets[0:1000, :] #  detection'box threshold
    return dets


# def draw_bboxes(det, img, prob,j):
def draw_bboxes(det, img, prob,filename='11_01.jpg'):
    num = 0
    xmin,xmax,ymin,ymax=0,0,0,0
    for i in range(det.shape[0]):
        xmin = int(det[i][0])
        xmax = int(det[i][2])
        ymin = int(det[i][1])
        ymax = int(det[i][3])
        score = det[i][4]
        if score < float(prob):
            continue
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        num = num + 1
        #print('xmin,ymin,xmax,ymax = ', xmin, ymin, xmax, ymax)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)

    # for video
    # print('j=',j)
    # cv2.imwrite(os.path.join(pic_path,'{}.jpg'.format(j)),img)
    # cv2.waitKey(1)
    # cv2.imshow('frame',img)

    #for images
    print('num = ',num)
    cv2.imwrite(filename, img)
    cv2.waitKey(5000)

# pic_path = '/home/zhex/test_result/hua01_03'
# video_path = '/home/zhex/Videos/test_video/hua01/03.mp4'
#
# def main():
#     vid = cv2.VideoCapture(video_path)
#     if not vid.isOpened():
#         raise IOError("couldn't open the video")
#     j = 0
#     while True:
#         j = j+1
#         return_value,frame = vid.read()
#         if not return_value:
#             break
#         height, width = frame.shape[:2] # 1080,1920
#         # image = cv2.resize(frame, (width*2, height*24), interpolation=cv2.INTER_CUBIC) # 8ms 2160 3840
#         image = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)
#         # print('image_shape = ',image.shape) # image_shape = (540,960) -> 416*416
#         # image = frame
#         max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5  # 1.1377777775128681
#         max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
#         shrink = max_im_shrink if max_im_shrink < 1 else 1
#         # print('shrink = ',shrink) # 1
#
#         # 开始计时时刻
#         st = cv2.getTickCount()
#         # multi-scale
#         det0 = detect_face(image, shrink)                     # origin test
#         det1 = flip_test(image, shrink)                       # flip test
#         # print('det1_shape = ',det1.shape) # 82*5
#
#         [det2, det3] = multi_scale_test(image,max_im_shrink)  # multi_scale_test
#         # print('det2_shape = ',det2.shape) # 76*5
#         # print('det3_shape = ',det3.shape) # 0*5
#
#         det4 = multi_scale_test_pyramid(image, max_im_shrink) # multi_scale_test_pyramid
#         # print('det4_shape = ',det4.shape) # 10*5
#
#         det = np.row_stack((det0, det1, det2, det3, det4))    # merged by row
#         # det = np.row_stack((det0, det1))
#         # print('det_shape =',det.shape) # 250*5
#         dets = bbox_vote(det)
#
#         # dets[:, 0] = dets[:, 0] / 2
#         # dets[:, 1] = dets[:, 1] / 2
#         # dets[:, 2] = dets[:, 2] / 2
#         # dets[:, 3] = dets[:, 3] / 2
#
#         draw_bboxes(dets, image, args.probability,j)
#         # draw_bboxes(dets, frame, args.probability)
#         # 终止计时时刻
#         et = cv2.getTickCount()
#         tt = 1000*(et - st)/cv2.getTickFrequency()
#         print('Inference time = ',tt)

# img_path = '/home/zhex/test_result/shengyang/test/11.jpg'
def test(dirpath="./"):
    for root, _, files in os.walk(dirpath): # for root, dirs, files in os.walk(path):
        pic_list = [f for f in files if f.split(".")[-1].lower() in ['jpg', 'png']]
        for one_pic in pic_list:
            abs_path = os.path.join(root, one_pic)
            img = cv2.imread(abs_path)

            image = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
            max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5  # 1.1377777775128681
            print('max_im_shrink= ', max_im_shrink)
            max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
            shrink = max_im_shrink if max_im_shrink < 1 else 1

            # start_time
            st = cv2.getTickCount()
            det0 = detect_face(image, shrink)
            det1 = flip_test(image, shrink)
            # 多尺度金字塔测试
            [det2, det3] = multi_scale_test(image, max_im_shrink)
            det4 = multi_scale_test_pyramid(image, max_im_shrink)
            det = np.row_stack((det0, det1, det2, det3))
            dets = bbox_vote(det)

            out_pic = os.path.join(root, "out_" + one_pic)
            draw_bboxes(dets, image, args.probability, out_pic)

img_path = 'test_image/u01.jpg'
def main():
    print(img_path)
    img = cv2.imread(img_path)
    print('img0_shape = ',img.shape)
    height,width = img.shape[:2]
    #image = cv2.resize(img, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(img, (640, 640), interpolation=cv2.INTER_CUBIC)
    max_im_shrink = (0x7fffffff / 200.0 / (image.shape[0] * image.shape[1])) ** 0.5  # 1.1377777775128681
    print('max_im_shrink= ',max_im_shrink)
    max_im_shrink = 3 if max_im_shrink > 3 else max_im_shrink
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    # start_time
    st = cv2.getTickCount()
    det0 = detect_face(image, shrink)
    # det1 = flip_test(image, shrink)
    # 多尺度金字塔测试
    # [det2, det3] = multi_scale_test(image, max_im_shrink)
    # det4 = multi_scale_test_pyramid(image, max_im_shrink)
    # det = np.row_stack((det0, det1, det2, det3))
    dets = bbox_vote(det0)
    filename = 'out_'+os.path.basename(img_path)
    draw_bboxes(dets, image, args.probability,filename)

    # end_time
    et = cv2.getTickCount()
    tt = 1000 * (et - st) / cv2.getTickFrequency()
    print('Inference time = ', tt)

main()
# test("test_image")
