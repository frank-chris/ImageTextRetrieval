import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
from models.model import Model
from directory import check_file
import random
import numpy as np
    
    
def network_config(args, split='train', param=None, resume=False, model_path=None, ema=False):
    network = Model(args)
    network = nn.DataParallel(network).cuda()
    cudnn.benchmark = True
    args.start_epoch = 0

    # process network params
    if resume:
        check_file(model_path, 'model_file')
        checkpoint = torch.load(model_path)
        args.start_epoch = checkpoint['epoch'] + 1
        # best_prec1 = checkpoint['best_prec1']
        #network.load_state_dict(checkpoint['state_dict'])
        network_dict = checkpoint['network']
        # if ema:
        #     logging.info('==> EMA Loading')
        #     network_dict.update(checkpoint['network_ema'])
        network.load_state_dict(network_dict) 
        print('==> Loading checkpoint "{}"'.format(model_path))
    else:
        # pretrained
        if model_path is not None:
            print('==> Loading from pretrained models')
            network_dict = network.state_dict()
            if args.image_model == 'mobilenet_v1':
                cnn_pretrained = torch.load(model_path)['state_dict']
                start = 7
            else:
                cnn_pretrained = torch.load(model_path)
                start = 0
            # process keyword of pretrained model
            prefix = 'module.image_model.'
            pretrained_dict = {prefix + k[start:] :v for k,v in cnn_pretrained.items()}
            pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in network_dict}
            network_dict.update(pretrained_dict)
            network.load_state_dict(network_dict)

    # process optimizer params
    if split == 'test':
        optimizer = None
    else:
        # optimizer
        # different params for different part
        cnn_params = list(map(id, network.module.image_model.parameters()))
        other_params = filter(lambda p: id(p) not in cnn_params, network.parameters())
        other_params = list(other_params)
        if param is not None:
            other_params.extend(list(param))
        param_groups = [{'params':other_params},
            {'params':network.module.image_model.parameters(), 'weight_decay':args.wd}]
        optimizer = torch.optim.Adam(
            param_groups,
            lr = args.lr, betas=(args.adam_alpha, args.adam_beta), eps=args.epsilon)
        if resume:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))
    # seed
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    return network, optimizer
