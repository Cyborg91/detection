# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pyramidBox.net.data import face as cfg
from ..box_utils import match, log_sum_exp, matchNoBipartite


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)

    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, bipartite=True,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.bipartite = bipartite
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,num_classes)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)                # 获取张量第0维尺寸,batch_size
        priors = priors[:loc_data.size(1), :] # 获取张量第1维尺寸,num_priors
        num_priors = (priors.size(0))         # 获取张量第1维尺寸,num_priors
        num_classes = self.num_classes        # 获取类别数量


        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            # tensor.data tensor.detach只取出tensor里面的数据,舍弃其他反向传播所保存的信息
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data

            '''
            def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
                """Match each prior box with the ground truth box of the highest jaccard
                overlap, encode the bounding boxes, then return the matched indices
                corresponding to both confidence and location preds.
                Args:
                    threshold: (float) The overlap threshold used when mathing boxes.
                    truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
                    priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
                    variances: (tensor) Variances corresponding to each prior coord,
                        Shape: [num_priors, 4].
                    labels: (tensor) All the class labels for the image, Shape: [num_obj].
                    loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
                    conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
                    idx: (int) current batch index
                Return:
                    The matched indices corresponding to 1)location and 2)confidence preds.
                """
            def matchNoBipartite(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
                """Match each prior box with the ground truth box of the highest jaccard
                overlap, encode the bounding boxes, then return the matched indices
                corresponding to both confidence and location preds.
                Args:
                    threshold: (float) The overlap threshold used when mathing boxes.
                    truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
                    priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
                    variances: (tensor) Variances corresponding to each prior coord,
                        Shape: [num_priors, 4].
                    labels: (tensor) All the class labels for the image, Shape: [num_obj].
                    loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
                    conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
                    idx: (int) current batch index
                Return:
                    The matched indices corresponding to 1)location and 2)confidence preds.
                """
            '''

            if self.bipartite: # True
                match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            else:
                matchNoBipartite(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)


        pos = conf_t > 0
        #pos = Variable(pos, requires_grad=False)
        num_pos = pos.sum(dim=1, keepdim=True) # x.sum(0) 按列求和 x.sum(1)按行求和

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # tensor.unsqueeze(a,0) 按行进行升维度
        # tensor.unsqueeze(a,1) 按列进行升维
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1,1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        if N==0:
            N = num
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
