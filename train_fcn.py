import logging
import os.path
from collections import deque

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable
import torch.utils.data as data

from cycada.data.data_loader import get_fcn_dataset as get_dataset
from cycada.models import get_model
from cycada.models.models import models
from cycada.transforms import augment_collate
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.tools.util import make_variable


class AddaDataset(data.Dataset):

    def __init__(self, im, label):
        self.im = im # [8400, 3, 256, 256]
        self.label = label # [8400, 256, 256]
        

    def __getitem__(self, index):
        return self.im[index], self.label[index]

    def __len__(self):
        return len(self.im)

dir="/home/liuxina/Downloads/cycada_release/data_npy/"

def load_data(dset, batch=64, kwargs={}):
    is_train = (dset == 'train')
    
    file=dir+'mr_img.npy'
    im = np.load(file)
    im = torch.from_numpy(im)
    im = im.type(torch.FloatTensor)
    print(im.shape)

    file=dir+'mr_gt.npy'
    label = np.load(file)
    label = torch.from_numpy(label)
    label = label.type(torch.LongTensor)
    print(label.shape)
    
    dataset = AddaDataset(im, label)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, 
            shuffle=is_train, **kwargs)      

    return loader

def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))


def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss2d(weight=weights, size_average=True, 
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score), label)
    return loss
 

@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--step', type=int)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=720)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--num_cls', default=19, type=int)
@click.option('--gpu', default='0')
def main(output, dataset, datadir, batch_size, lr, step, iterations, 
        momentum, snapshot, downscale, augmentation, fyu, crop_size, 
        weights, model, gpu, num_cls):
    if weights is not None:
        raise RuntimeError("weights don't work because eric is bad at coding")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()
    
    logdir = 'runs/{:s}/{:s}'.format(model, '-'.join(dataset))
    writer = SummaryWriter(log_dir=logdir)
    net = get_model(model, num_cls=num_cls, pretrained=False)
    net.cuda()
    transform = []
    target_transform = []
    if downscale is not None:
        transform.append(torchvision.transforms.Scale(1024 // downscale))
        target_transform.append(
            torchvision.transforms.Scale(1024 // downscale,
                                         interpolation=Image.NEAREST))
    transform.extend([
        torchvision.transforms.Scale(1024),
        net.transform
        ])
    target_transform.extend([
        torchvision.transforms.Scale(1024, interpolation=Image.NEAREST),
        to_tensor_raw
        ])
    transform = torchvision.transforms.Compose(transform)
    target_transform = torchvision.transforms.Compose(target_transform)
    

    datasets = [get_dataset(name, os.path.join(datadir,name), transform=transform,
                            target_transform=target_transform)
                for name in dataset]

    if weights is not None:
        weights = np.loadtxt(weights)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=0.0005)

    if augmentation:
        collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
    loaders = load_data(dset="train",batch=batch_size)
    '''
    loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2,
                                           collate_fn=collate_fn,
                                           pin_memory=True)
               for dataset in datasets]
    '''
    iteration = 0
    losses = deque(maxlen=10)
    
    i = 0
    for i in range(10000):

        for im, label in loaders:
            # Clear out gradients
            opt.zero_grad()
            
            # load data/label
            im = make_variable(im, requires_grad=False)
            label = make_variable(label, requires_grad=False)
            
            # forward pass and compute loss
            preds = net(im)
            loss = supervised_loss(preds, label)
            
            # backward pass
            loss.backward()
            losses.append(loss.item())
            
            # step gradients
            opt.step()

                
            # log results
            #print('snapshot:',snapshot)
            #print('step:',step)
            if iteration % 10 == 0:
                logging.info('Iteration {}:\t{}'
                                .format(iteration, np.mean(losses)))
                writer.add_scalar('loss', np.mean(losses), iteration)
            iteration += 1
            if step is not None and iteration % step == 0:
                logging.info('Decreasing learning rate by 0.1.')
                step_lr(optimizer, 0.1)
            if iteration % snapshot == 0:
                torch.save(net.state_dict(),
                            '{}-iter{}.pth'.format(output, iteration))
            if iteration >= iterations:
                logging.info('Optimization complete.')
                break
        
        i = i+1
    

'''
    for im, label in roundrobin_infinite(*loaders):
        # Clear out gradients
        opt.zero_grad()
        
        # load data/label
        im = make_variable(im, requires_grad=False)
        label = make_variable(label, requires_grad=False)
        
        # forward pass and compute loss
        preds = net(im)
        loss = supervised_loss(preds, label)
        
        # backward pass
        loss.backward()
        losses.append(loss.item())
        
        # step gradients
        opt.step()

            
        # log results
        #print('snapshot:',snapshot)
        #print('step:',step)
        if iteration % 10 == 0:
            logging.info('Iteration {}:\t{}'
                            .format(iteration, np.mean(losses)))
            writer.add_scalar('loss', np.mean(losses), iteration)
        iteration += 1
        if step is not None and iteration % step == 0:
            logging.info('Decreasing learning rate by 0.1.')
            step_lr(optimizer, 0.1)
        if iteration % snapshot == 0:
            torch.save(net.state_dict(),
                        '{}-iter{}.pth'.format(output, iteration))
        if iteration >= iterations:
            logging.info('Optimization complete.')
            break
'''                

if __name__ == '__main__':
    main()