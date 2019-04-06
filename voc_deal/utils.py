import csv
import os

def read_csv(csv_path, pre_dir):
    '''
    :param csv_path:csv文件路径
    :param pre_dir: 图片数据所在的文件夹
    :return:
    '''
    label_dict = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = True
        for line in reader:
            # 除去文件头
            if header:
                header = False
                continue
            # 处理文件存储路径，当做标签
            image_path = os.path.join(pre_dir, line[0])
            # 处理后面的bbox
            bbox = []

            if line[1] is not None and len(line[1].strip()) > 0:
                for i in line[1].split(';'):
                    if i is not None and len(i.strip()) > 0:
                        bbox.append(list(map(lambda x: round(float(x.strip())), i.split(' '))))
            # 添加到label_dict
            label_dict.setdefault(image_path, bbox)
    return label_dict
   
