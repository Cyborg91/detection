#!/usr/bin/python3.6
# encoding: utf-8

import utils
import os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def save_xml(image_name, bbox, save_dir='/home/zhex/data/yuncong/Annotations', width=640, height=480, channel=3):
    '''
  CSV file one column
  xxxxxx.jpg [[1,2,3,4],...]
  change into
  xxxxxx.xml

  :param image_name
  :param bbox:
  :param save_dir
  :param width
  :param height
  :param channel
  :return:
  '''

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'JPEGImages'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel

    for x, y, w, h in bbox:
        left, top, right, bottom = x, y, x + w, y + h
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'car'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    save_xml = os.path.join(save_dir, image_name.replace('jpg', 'xml'))
    with open(save_xml, 'wb') as f:
        f.write(xml)

    return


def change2xml(label_dict={}):
    for image in label_dict.keys():
        image_name = os.path.split(image)[-1]
        bbox = label_dict.get(image, [])
        save_xml(image_name, bbox)
    return


def main():
    label_dict = utils.read_csv(csv_path=r'train.csv',pre_dir=r'')
    change2xml(label_dict)

main()
