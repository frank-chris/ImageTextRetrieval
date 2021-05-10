import os
import sys
import shutil
import time
import random
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms

from utils.metric import Loss
from utils.helpers import avg_calculator
from utils.directory import makedir, check_file

from datasets.fashion import Fashion
from models.model import Model

from train_params import get_train_args


"""
Train for one epoch, apply backpropagation, return loss after current epoch
"""
def train(epoch, train_loader, network, optimizer, compute_loss, args):

    # Trains for 1 epoch
    batch_time = avg_calculator()
    train_loss = avg_calculator()

    #switch to train mode, req in some modules 
    network.train()

    end = time.time()

    for step, (images, captions, labels, captions_length) in enumerate(train_loader):

        images = images
        labels = labels
        captions = captions

        # compute loss
        image_embeddings, text_embeddings = network(images, captions, captions_length)
        cmpm_loss, pos_avg_sim, neg_arg_sim = compute_loss(image_embeddings, text_embeddings, labels)
        

        if step % 10 == 0:
            print('epoch:{}, step:{}, cmpm_loss:{:.3f}'.format(epoch, step, cmpm_loss))
        

        # compute gradient and do ADAM step
        optimizer.zero_grad()
        cmpm_loss.backward()
        #nn.utils.clip_grad_norm(network.parameters(), 5)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        train_loss.update(cmpm_loss, images.shape[0])
                
    return train_loss.avg, batch_time.avg



"""
Initialise the data loader
"""
def get_data_loader(image_dir, anno_dir, batch_size, split, max_length):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_split = Fashion(image_dir, anno_dir, split, max_length, train_transform)

    loader = data.DataLoader(data_split, batch_size, shuffle=True, num_workers=2)

    return loader


"""
Initialise the network object
"""
def get_network(args, resume=False, model_path=None):

    network = Model(args)
    network = nn.DataParallel(network)
    # cudnn.benchmark = True
    args.start_epoch = 0

    # process network params
    if resume:
        if model_path==None:
            raise ValueError('Supply the model path with --model_path while using ---resume')
        check_file(model_path, 'model_file')
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch'] + 1
        network_dict = checkpoint['network']
        network.load_state_dict(network_dict) 
        print('==> Loading checkpoint "{}"'.format(model_path))

    return network


"""
Initialise optimizer object
"""
def get_optimizer(args, network=None, param=None, resume=False, model_path=None):

    #process optimizer params

    # optimizer
    # different params for different part
    cnn_params = list(map(id, network.module.image_model.parameters()))
    other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
    other_params = list(other_params)
    if param is not None:
        other_params.extend(list(param))
    param_groups = [{'params':other_params}, {'params':network.module.image_model.parameters(), 'weight_decay':args.wd}]

    optimizer = torch.optim.Adam(
        param_groups,
        lr = args.lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon
    )

    if resume:
        check_file(model_path, 'model_file')
        checkpoint = torch.load(model_path)
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))
    # seed
    
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # torch.cuda.manual_seed_all(manualSeed)

    return optimizer


"""
Modify learning rate
"""
def adjust_lr(optimizer, epoch, args):

    # Decay learning rate by args.lr_decay_ratio every args.epoches_decay

    if args.lr_decay_type == 'exponential':
        
        lr = args.lr * (1 - args.lr_decay_ratio)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


"""
Multistep Learning Rate Scheduler which decays learning rate after a no. of epochs specified by the user
"""
def lr_scheduler(optimizer, args):

    if '_' in args.epoches_decay:
        epoches_list = args.epoches_decay.split('_')
        epoches_list = [int(e) for e in epoches_list]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args.epoches_decay))
        
    return scheduler


"""
Saves the checkpoints
"""
def save_checkpoint(state, epoch, ckpt_dir, is_best):

    filename = os.path.join(ckpt_dir, str(args.start_epoch + epoch)) + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        dst_best = os.path.join(ckpt_dir, 'model_best', str(epoch)) + '.pth.tar'
        shutil.copyfile(filename, dst_best)




"""
Initializes data loader, loss object, network, optimizer, runs the desired no. of epochs
"""
def main(args):
    
    train_loader = get_data_loader(args.image_dir, args.anno_dir, args.batch_size, 'train', args.max_length)
    
    # loss
    compute_loss = Loss(args)
    nn.DataParallel(compute_loss)
    
    # network
    network = get_network(args, args.resume, args.model_path)
    optimizer = get_optimizer(args, network, compute_loss.parameters(), args.resume, args.model_path)
    
    # lr_scheduler
    scheduler = lr_scheduler(optimizer, args)

    for epoch in range(args.num_epoches - args.start_epoch):
        # train for one epoch
        train_loss, train_time = train(args.start_epoch + epoch, train_loader, network, optimizer, compute_loss, args)
        # evaluate on validation set
        print('Train done for epoch-{}'.format(args.start_epoch + epoch))

        
        state = {'network': network.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'W': compute_loss.W, 
                    'epoch': args.start_epoch + epoch
                }
        
        save_checkpoint(state, epoch, args.checkpoint_dir, False)

        adjust_lr(optimizer, args.start_epoch + epoch, args)
        scheduler.step()


        for param in optimizer.param_groups:
            print('lr:{}'.format(param['lr']))
            break




if __name__ == "__main__":

    # Get arguments passed by user
    args = get_train_args()

    # Validate existence of image and annotation directory
    if not os.path.exists(args.image_dir):
        raise ValueError('Supply the dataset directory with --image_dir')
    if not os.path.exists(args.anno_dir):
        raise ValueError('Supply the anno file with --anno_dir')

    # save checkpoint
    makedir(args.checkpoint_dir)
    makedir(os.path.join(args.checkpoint_dir,'model_best'))

    main(args)