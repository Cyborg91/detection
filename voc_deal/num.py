#!/usr/bin/python3.6
# encoding: utf-8

import numpy as np
filex = open('yuncong.txt','r')
filey = open('p.txt','a')
for line in filex.readlines():
    line = line.strip().split(' ')
    img_path = line[0]
    bbox = line[1:]
    bbox = np.array(list(map(int,boxes)))
    bbox = bbox.reshape((-1,4))
    new_line = img_path + ','
    for temp in bbox:
        new_line = new_line + str(temp[0])+ ' '+ str(temp[1])+ ' '+str(temp[2])+ ' ' +str(temp[3])+ ';'
    filey.write(newline+'\n')
