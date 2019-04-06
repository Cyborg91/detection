import utils
import csv
def write_csv(result_dict, out_path='zhe.csv'):
    with open(out_path, 'w', newline='') as f:

        for image in result_dict.keys():
            writer = csv.writer(f)
            # image_name = os.path.split(image)[-1]
            bbox = result_dict.get(image, [])
            bbox_rs = ';'.join([' '.join(str(int(id)) for id in i) for i in bbox])
            writer.writerow([image, bbox_rs])

if __name__ == '__main__':
    label_dict = utils.read_csv(csv_path=r'/home/zhex/tools/head_detect/yuncong/train.csv',pre_dir=r'')
    write_csv(label_dict)
