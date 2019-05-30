import torch as t
from skimage import transform as sktsf
import cv2
from torchvision import transforms as tvtsf
from . import util
import numpy as np
from src.config import opt
from PIL import Image


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def pytorch_normalze(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def caffe_normalize(img):
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

def preprocess(img, min_size=416, max_size=416):
    """ Function to preprocess the input image. 
    
    Scales the input image in such a manner that the shorter side of the 
    image is converted to the size equal to min_size. 
    Also normalizes the input image. 

    Args: 
        img: Input image that is to be preprocessed. 
        min_size: size to which the smaller side of the image is to be 
                    converted. 
        max_size: size to which the larger side of the image is to be 
                    converted. 
    """
    # C, H, W = img.shape
    # scale1 = min_size / min(H, W)
    # scale2 = max_size / max(H, W)
    # scale = min(scale1, scale2)
    # print('scale = ',scale)
    img = img / 255
    # print('shape_img =',img.shape)
    #
    # H = int(H * scale) #234
    # W = int(W * scale) #416
    # print('H , W = ',H,W)

    # st = cv2.getTickCount()
    # # resize_time = 118ms 时间主要消耗在这里
    # img = sktsf.resize(img, (C, H , W ), mode='reflect')
    # print('zhex_shape =',img.shape)
    # et = cv2.getTickCount()
    # tt = (et-st)/cv2.getTickFrequency()
    # print('prerocess_time = ',1000*tt)

    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)

class Transform(object):
    """ Class to transform the image given as input. 
    """
    # def __init__(self, min_size=600, max_size=1000):
    def __init__(self, min_size=416, max_size=416):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bboxs, _ = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bboxs = util.resize_bbox(bboxs, (H, W), (o_H, o_W))

        return img, bboxs, scale

class HeadDataset:
    def __init__(self, dl):
        self.datalist = dl

    def get_example(self, idx):
        """ Read the image from the image path, specific to the idx given 
            argument to the function. 

        Args: 
            idx: idx of the image to be read. 
        
        Returns: 
            img: image after reading from the path
            bboxs: ground-truth corresponding to the image. 
            n_bboxs: number of heads in the image. 
        """
        data_obj = self.datalist[idx]
        img_path = data_obj.path
        n_boxs = data_obj.n_boxs
        bboxs = data_obj.bboxs
        img = self.read_image(img_path)
        return img, bboxs, n_boxs

    def read_image(self, path, dtype=np.float32):
        f = Image.open(path)
  
        # Convert to RGB
        f.convert('RGB')
        # Convert to a numpy array
        img = np.asarray(f, dtype=np.float32)
        # Transpose the final image array i.e. C, H, W
        return img.transpose((2, 0, 1))

# class ShanghaiTechDataset:
#     def __init__(self, dl):
#         self.datalist = dl

#     def get_example(self, idx):
#         """ Read the image from the image path, specific to the idx given 
#             argument to the function. 

#         Args: 
#             idx: idx of the image to be read. 
        
#         Returns: 
#             img: image after reading from the path
#             bboxs: ground-truth corresponding to the image. 
#             n_bboxs: number of heads in the image. 
#         """
#         data_obj = self.datalist[idx]
#         img_path = data_obj.path
#         print img_path
#         n_boxs = data_obj.n_boxs
#         bboxs = data_obj.bboxs
#         img = self.read_image(img_path)
#         return img, bboxs, n_boxs

#     def read_image(self, path, dtype=np.float32):
#         f = Image.open(path)
#         f = f.resize((640,480), Image.ANTIALIAS)
#         # Convert to RGB
#         f.convert('RGB')
#         # Convert to a numpy array
#         img = np.asarray(f, dtype=np.float32)
#         # Transpose the final image array i.e. C, H, W
#         return img.transpose((2, 0, 1))    

# class ShanghaiTechTestDataset:
#     def __init__(self, datalist):
#         self.datalist = datalist
#         self.db = ShanghaiTechDataset(datalist)
#         self.tsf = Transform(opt.min_size, opt.max_size)
    
#     def __getitem__(self, idx):
#         ori_img, bboxs, n_boxs = self.db.get_example(idx)  
#         img, bboxs, scale = self.tsf((ori_img, bboxs, n_boxs))
#         return img.copy(), bboxs.copy(), scale     

#     def __len__(self):
#         return len(self.datalist)

class Dataset:
    """Dataset class which is assigned by the datalist of the dataset,
    Containes member functions to transform the image and preprocess the 
    input image. This class is accessed by the data loader during the 
    training and the testing phase. 

    On calling the object it returns an image, ground-truth annotations 
    and the scale which is required to transform the image. 

    Args: 
        datalist : list of data object where the number of items in the 
                    list is equal to the number of datapoints in the split.

    Returns: 

    """
    def __init__(self, datalist):
        self.datalist = datalist
        self.db = HeadDataset(datalist)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        """__getitem__ function that reads the input image corresponding 
            to a idx. After reading the input image, transforms the image
            with the pre-defined set of operations. 

        Args: 
            idx: Index of the image to be read from the data list. 
        
        Returns: 
            img: The transform (normalized) image. 
            bboxs: The ground-truth annotations. 
            scale: scale used to rescale the image.
        """
        ori_img, bboxs, n_boxs = self.db.get_example(idx)  
        img, bboxs, scale = self.tsf((ori_img, bboxs, n_boxs))
        return img.copy(), bboxs.copy(), scale
        # return ori_img, bboxs, n_boxs
    def __call__(self, idx):
        ori_img, bboxs, n_boxs = self.db.get_example(idx)
        return ori_img, bboxs, n_boxs

    def __len__(self):
        return len(self.datalist)




