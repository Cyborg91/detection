# -*- coding: utf-8 -*-
# crop affine channel_shuffle blur salt_and_pepper

f = open('/home/zhex/data/yuncong/our_train.txt','r')
lines = f.readlines()

f_origin = open('our_origin.txt','a')
i = 0
count0 = 0
while i < len(lines):
    img_id = lines[i].rstrip().split(' ')[0]
    #f_origin.write('"{}": '.format(img_id))
    f_origin.write('myaug/our/'+img_id + '\n')

    num = int(lines[i].rstrip().split(' ')[1])
    print('num = ',num)
    f_origin.write('{}\n'.format(num))

    if num == 0:
        count0 = count0 + 1

    coordinations = []
    j = 1 # 控制每一行中坐标的读取

    for box_idx in range(num):
        # 取每一行中标记的坐标(x,y,w,h)
        coordinations = lines[i].rstrip().split(' ')[2:]
        x, y, w, h = map(int,coordinations[j:j+4])
        f_origin.write('{} {} {} {} 0 0 0 0 0\n'.format(x, y, x+w, y+h))
        j =j + 5

    i = i+1





