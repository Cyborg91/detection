import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
import time
from cannet import CANNet
from my_dataset import CrowdDataset

def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    model=CANNet()
    model.load_state_dict(torch.load(model_param_path))
    model.cuda()
    dataset=CrowdDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    mae=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(tqdm(dataloader)):
            img=img.cuda()
            gt_dmap=gt_dmap.cuda()
            # forward propagation
            et_dmap=model(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,gt_dmap_root,model_param_path):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    model=CANNet().cuda()
    model.load_state_dict(torch.load(model_param_path))
    model.eval()
    dataset=CrowdDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    print('dataloader = ',dataloader)

    for img,gt_dmap,img_name in dataloader:
        print('img_shape = ',img.shape)
        st = time.time()
        img=img.cuda()
        gt_dmap=gt_dmap.cuda()
        # forward propagation
        et_dmap=model(img).detach()
        et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
        pred_frame = plt.gca()
        plt.imshow(et_dmap,cmap=CM.jet)
        plt.show()
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False)
        pred_frame.spines['bottom'].set_visible(False)
        pred_frame.spines['left'].set_visible(False)
        pred_frame.spines['right'].set_visible(False)
        plt.savefig( 'result/'+str(img_name)+'_out' + '.png',bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        print('tt = ',time.time()-st)
        break

if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root='./datasets/part_A_final/test_data/images'
    gt_dmap_root='./datasets/part_A_final/test_data/ground_truth'
    model_param_path='./checkpoints/cvpr2019CAN_353model.pth'
    st = time.time()
    estimate_density_map(img_root,gt_dmap_root,model_param_path)
    et = time.time()
    print('tt = ',et-st)