import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from statistics import median


def pairwise_distance(A, B):

    A_square = torch.sum(A * A, dim=1, keepdim=True)
    B_square = torch.sum(B * B, dim=1, keepdim=True)

    distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())

    return distance


def one_hot_coding(index, k):
    if type(index) is torch.Tensor:
        length = len(index)
    else:
        length = 1
    out = torch.zeros((length, k), dtype=torch.int64).cuda()
    index = index.reshape((len(index), 1))
    out.scatter_(1, index, 1)
    return out



"""
LOSS MODULE
"""


class Loss(nn.Module):

    def __init__(self, args):

        super(Loss, self).__init__()

        self.CMPM = True
        self.epsilon = args.epsilon

        self.num_images = args.num_images

        if args.resume:
            checkpoint = torch.load(args.model_path)
            self.W = Parameter(checkpoint['W'])
            print('=====> Loading weights from pretrained path')
        else:
            self.W = Parameter(torch.randn(args.feature_size, args.num_images))
            nn.init.xavier_uniform_(self.W.data, gain=1)        


    # CMPM Loss
    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)
        
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)
         
        i2t_pred = F.softmax(image_proj_text, dim=1)
        #i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred.to(device="cpu") * (F.log_softmax(image_proj_text.to(device="cpu"), dim=1) - torch.log(labels_mask_norm.to(device="cpu") + self.epsilon))
        
        t2i_pred = F.softmax(text_proj_image, dim=1)
        #t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred.to(device="cpu") * (F.log_softmax(text_proj_image.to(device="cpu"), dim=1) - torch.log(labels_mask_norm.to(device="cpu") + self.epsilon))

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))
        
        return cmpm_loss, pos_avg_sim, neg_avg_sim


    def forward(self, image_embeddings, text_embeddings, labels):

        cmpm_loss = 0.0
        neg_avg_sim = 0.0
        pos_avg_sim =0.0

        if self.CMPM:
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(image_embeddings, text_embeddings, labels)
        
        return cmpm_loss, pos_avg_sim, neg_avg_sim

        
""""
Recall rate and Median Rank
"""

# Computes the recall rate
def compute_topk(query, gallery, target_query, target_gallery, k=[1,5,10], reverse=False):
    result = []
    query = query / query.norm(dim=1,keepdim=True)
    gallery = gallery / gallery.norm(dim=1,keepdim=True)
    sim_cosine = torch.matmul(query, gallery.t())
    result.extend(topk(sim_cosine, target_gallery, target_query, k=[1,5,10]))
    if reverse:
        result.extend(topk(sim_cosine, target_query, target_gallery, k=[1,5,10], dim=0))
    return result


def topk(sim, target_gallery, target_query, k=[1,5,10], dim=1):
    result = []
    maxk = max(k)
    size_total = len(target_gallery)
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    if dim == 1:
        pred_labels = pred_labels.t()
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

    for topk in k:
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result


"""
Computes the Median Rank
"""
def compute_mr(query, gallery, target_query, target_gallery, k, reverse=False):
    result = []
    query = query / query.norm(dim=1,keepdim=True)
    gallery = gallery / gallery.norm(dim=1,keepdim=True)
    sim_cosine = torch.matmul(query, gallery.t())
    result.extend(mr(sim_cosine, target_gallery, target_query, k))
    if reverse:
        result.extend(mr(sim_cosine, target_query, target_gallery, k, dim=0))
    return result


def mr(sim, target_gallery, target_query, k, dim=1):
    result = []
    maxk = k
    size_total = len(target_gallery)
    _, pred_index = sim.topk(maxk, dim, True, True)
    pred_labels = target_gallery[pred_index]
    if dim == 1:
        pred_labels = pred_labels.t()
    correct = pred_labels.eq(target_query.view(1,-1).expand_as(pred_labels))

    ranks = []
    for row in correct.t():
        temp = torch.where(row > 0)[0]
        if temp.shape[0] > 0:
            ranks.append(temp[0].item()  + 1)
        else:
            ranks.append(k)
            # print('incr. k')

    result.append(median(ranks) * 100 / size_total)
    return result