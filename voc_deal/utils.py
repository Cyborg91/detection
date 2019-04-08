import csv
import os

def read_csv(csv_path, pre_dir):
    '''
    :param csv_path:csv file_path
    :param pre_dir: pic_dir
    :return:
    '''
    label_dict = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = True
        for line in reader:
            # delete file head
            if header:
                header = False
                continue
            
            image_path = os.path.join(pre_dir, line[0])
            bbox = []

            if line[1] is not None and len(line[1].strip()) > 0:
                for i in line[1].split(';'):
                    if i is not None and len(i.strip()) > 0:
                        bbox.append(list(map(lambda x: round(float(x.strip())), i.split(' '))))
            # add to label_dict
            label_dict.setdefault(image_path, bbox)
    return label_dict

def write_csv(result_dict, out_path='zhe.csv'):
    '''
    :param result_dict: 
    :param out_path:
    :return:
    '''
    with open(out_path, 'w', newline='') as f:

        for image in result_dict.keys():
            writer = csv.writer(f)
            image_name = os.path.split(image)[-1]
            bbox = result_dict.get(image, [])
            bbox_rs = ';'.join([' '.join(str(int(id)) for id in i) for i in bbox])
            writer.writerow([image, bbox_rs])

if __name__ == '__main__':
    label_dict = read_csv(csv_path=r'./train.csv',pre_dir=r'/home/zhex/data')
    write_csv(label_dict)
   
