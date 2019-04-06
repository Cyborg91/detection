#!/usr/bin/python3.6
# encoding: utf-8

import xml.etree.ElementTree as ET
from os import getcwd

sets=[('yuncong', 'train'), ('yuncong', 'val'), ('yuncong', 'test')]

classes = ["head"]
def convert_annotation(image_id, list_file):
    in_file = open('/home/zhex/data/yuncong/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for image_set in sets:
    #image_ids = open('/home/zhex/data/yuncong/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    image_ids = open('/home/zhex/data/yuncong/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('/home/zhex/data/yuncong/JPEGImages/%s.jpg'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
