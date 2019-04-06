#!/usr/bin/python3.6
# encoding: utf-8


import os
import utils

def rename_image(label_dict={}, out_file='train_new.csv'):
    '''
    :param label_dict:
    :param out_file:
    :return:
    '''
    new_label_dict = {}
    i = 1
    with open(out_file, 'w') as f:
        for key in label_dict.keys():
            if not os.path.isfile(key):
                continue

            image_name = os.path.split(key)[-1]
            new_image_name = '%09d'%i + '.jpg'
            i = i + 1
            # 改名
            new_key = key.replace(image_name, new_image_name)
            os.renames(key, new_key)

            new_label_dict.setdefault(new_key, label_dict.get(key, []))
            utils.write_csv(new_label_dict, out_path=out_file)

    return out_file

if __name__ == '__main__':
    label_dict = utils.read_csv(csv_path=r'./train.csv',pre_dir=r'/home/zhex/data')
    rename_image(label_dict)

