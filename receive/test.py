import logging
import os
import os.path
from collections import deque
import itertools
from datetime import datetime

import scipy.misc

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torchvision import transforms, datasets

from PIL import Image
from torch.autograd import Variable

from cycada.models import get_model
from cycada.models.models import models
from cycada.models import VGG16_FCN8s, Discriminator
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.tools.util import make_variable


def check_label(label, num_cls):
    "Check that no labels are out of range"
    label_classes = np.unique(label.numpy().flatten())
    label_classes = label_classes[label_classes < 255]
    if len(label_classes) == 0:
        print('All ignore labels')
        return False
    class_too_large = label_classes.max() > num_cls
    if class_too_large or label_classes.min() < 0:
        print('Labels out of bound')
        print(label_classes)
        return False
    return True


class AddaDataset(data.Dataset):

    def __init__(self, mr_img, mr_gt, ct_img, ct_gt):
        self.mr_img = mr_img # [1024, 3, 256, 256]
        self.mr_gt = mr_gt # [8400, 256, 256]
        self.ct_img = ct_img
        self.ct_gt = ct_gt
        

    def __getitem__(self, index):
        return self.mr_img[index], self.ct_img[index], self.mr_gt[index], self.ct_gt[index]

    def __len__(self):
        return len(self.mr_img)


def forward_pass(net, discriminator, im, requires_grad=False, discrim_feat=False):
    if discrim_feat:
        score, feat = net(im)
        dis_score = discriminator(feat)
    else:
        score = net(im)
        dis_score = discriminator(score)
    if not requires_grad:
        score = Variable(score.data, requires_grad=False)
        
    return score, dis_score
#score四维 dis_score二维

# def dice(input, target):
#     N = target.size(0)
#     smooth = 1

#     input_flat = input.view(N, -1)
#     target_flat = target.view(N, -1)

#     intersection = input_flat * target_flat

#     print('intersection',intersection.sum(1))
#     print('input',input_flat.sum(1))
#     print('target',target_flat.sum(1))
#     loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)

#     print('loss',loss)

#     c = loss.sum()
#     loss = 1 - c.item () / N

#     print('diceloss',loss)

#     return loss

# class DiceLoss(nn.Module):
#     def __init__(self, input, target):
#         super(DiceLoss, self).__init__()
 
def DiceLoss(input, target):
    N = target.size(0)
    smooth = 1

    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)

    intersection = input_flat * target_flat

    loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
    loss = 1 - loss.sum() / N

    return loss


# class MulticlassDiceLoss(nn.Module):

# 	def __init__(self):
# 		super(MulticlassDiceLoss, self).__init__()
 
def MulticlassDiceLoss(input, target):

    C = target.shape[1]

    totalLoss = 0

    for i in range(C):
        diceLoss = DiceLoss(input, target[:,i])
        totalLoss += diceLoss

    totalLoss = totalLoss / C


    return totalLoss

# def Dice(inp, target, eps=1):
# 	# 抹平了，弄成一维的
#     input_flatten = inp.flatten()
#     target_flatten = target.flatten()
#     # 计算交集中的数量
#     overlap = np.sum(input_flatten * target_flatten)
#     # 返回值，让值在0和1之间波动
#     return np.clip(((2. * overlap) / (np.sum(target_flatten) + np.sum(input_flatten) + eps)), 1e-4, 0.9999)


dir="/home/chenxi/cycada_release_0717/data_npy_test/"

def load_data(dset, batch=64, kwargs={}):
    is_train = (dset == 'train')

    file=dir+'ct_gt.npy'
    ct_gt=np.load(file)
    ct_gt = torch.from_numpy(ct_gt)
    ct_gt = ct_gt.type(torch.LongTensor)


    file=dir+'mr_gt.npy'
    mr_gt=np.load(file)
    mr_gt = torch.from_numpy(mr_gt)
    mr_gt = mr_gt.type(torch.LongTensor)
 
 
    file=dir+'ct_img.npy'
    ct_img=np.load(file)
    ct_img = torch.from_numpy(ct_img)
    ct_img = ct_img.type(torch.FloatTensor)


    file=dir+'mr_img.npy'
    mr_img=np.load(file)
    mr_img = torch.from_numpy(mr_img)
    mr_img = mr_img.type(torch.FloatTensor)
    

    dataset = AddaDataset(mr_img, mr_gt, ct_img, ct_gt)


    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, 
            shuffle=True, **kwargs)
       

    return loader

@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--lr', '-l', default=0.0001)
@click.option('--momentum', '-m', default=0.9)
@click.option('--batch', default=64)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--crop_size', default=None, type=int)
@click.option('--half_crop', default=None)
@click.option('--cls_weights', type=click.Path(exists=True))
@click.option('--weights_discrim', type=click.Path(exists=True))
@click.option('--weights_init', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--lsgan/--no_lsgan', default=False)
@click.option('--num_cls', type=int, default=19)
@click.option('--gpu', default='0')
@click.option('--max_iter', default=10000)
@click.option('--lambda_d', default=1.0)
@click.option('--lambda_g', default=1.0)
@click.option('--train_discrim_only', default=False)
@click.option('--discrim_feat/--discrim_score', default=False)
@click.option('--weights_shared/--weights_unshared', default=False)


def main(output, dataset, datadir, lr, momentum, snapshot, downscale, cls_weights, gpu, 
        weights_init, num_cls, lsgan, max_iter, lambda_d, lambda_g,
        train_discrim_only, weights_discrim, crop_size, weights_shared,
        discrim_feat, half_crop, batch, model):
    
    # So data is sampled in consistent way
    np.random.seed(1337)
    torch.manual_seed(1337)
    logdir = 'runs/{:s}/{:s}_to_{:s}/lr{:.1g}_ld{:.2g}_lg{:.2g}'.format(model, dataset[0],
            dataset[1], lr, lambda_d, lambda_g)
    if weights_shared:
        logdir += '_weightshared'
    else:
        logdir += '_weightsunshared'
    if discrim_feat:
        logdir += '_discrimfeat'
    else:
        logdir += '_discrimscore'
    logdir += '/' + datetime.now().strftime('%Y_%b_%d-%H:%M')
    writer = SummaryWriter(log_dir=logdir)


    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()
    print('Train Discrim Only', train_discrim_only)
    
    net = get_model(model, num_cls=num_cls, pretrained=True, 
                weights_init=weights_init, output_last_ft=discrim_feat)
    '''
    net = get_model(model, num_cls=num_cls, pretrained=False, weights_init=None,
            output_last_ft=discrim_feat)
    '''
    net_src = net # shared weights

    odim = 1 if lsgan else 2
    idim = num_cls if not discrim_feat else 4096
    print('discrim_feat', discrim_feat, idim)
    print('discriminator init weights: ', weights_discrim)

    loader = load_data(dset="train",batch=batch)


    # Class weighted loss?
    if cls_weights is not None:
        weights = np.loadtxt(cls_weights)
    else:
        weights = None


    iteration = 0
    num_update_g = 0
    last_update_g = -1
    losses_super_s = deque(maxlen=100)
    losses_super_t = deque(maxlen=100)
    losses_dice = deque(maxlen=100)
    losses_rep = deque(maxlen=100)
    accuracies_dom = deque(maxlen=100)
    intersections = np.zeros([100,num_cls])
    unions = np.zeros([100, num_cls])
    accuracy = deque(maxlen=100)
    print('max iter:', max_iter)
   
    net.train()
    # discriminator.train()

    while iteration < max_iter:
        print('iteration:',iteration)
        
        for im_s, im_t, label_s, label_t in loader:
            # print(im_s.shape)
            
            if iteration > max_iter:
                break
           
            info_str = 'Iteration {}: '.format(iteration)
            
            if not check_label(label_s, num_cls):
                continue
            
            ###########################
            # 1. Setup Data Variables #
            ###########################
            im_s = make_variable(im_s, requires_grad=False)
            label_s = make_variable(label_s, requires_grad=False)
            im_t = make_variable(im_t, requires_grad=False)
            label_t = make_variable(label_t, requires_grad=False)

           
            #############################
            # 2. Optimize Discriminator #
            #############################
            
            score_t = Variable(net(im_t).data, requires_grad=False)

            print('iteration:',iteration)

            loss_dice = MulticlassDiceLoss(label_t, score_t)
            losses_dice.append(loss_dice.item())            

            info_str += " Dice:{:.3f} ".format(np.mean(losses_dice))
            writer.add_scalar('loss/dice', np.mean(losses_dice), iteration)

            if iteration % 1 == 0 and iteration > 0:
                logging.info(info_str)

            iteration += 1

    writer.close()


if __name__ == '__main__':
    main()
