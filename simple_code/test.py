import os
import sys
import time
import shutil
import gc
import torch
import torchvision.transforms as transforms

from utils.helpers import avg_calculator
from utils.metric import compute_topk, compute_mr
from utils.directory import makedir, check_file

from datasets.fashion import Fashion

# from config import data_config, network_config

from test_params import get_test_args


def get_metrics(test_loader, network, args):

    batch_time = avg_calculator()

    # switch to evaluate mode
    network.eval()

    num_samples = args.batch_size * len(test_loader)

    images_bank = torch.zeros((num_samples, args.feature_size))
    text_bank = torch.zeros((num_samples,args.feature_size))
    labels_bank = torch.zeros(num_samples)

    index = 0
    with torch.no_grad():

        timer = time.time()

        for images, captions, labels, captions_length in test_loader:

            test_images = images
            test_captions = captions

            image_embeddings, text_embeddings = network(test_images, test_captions, captions_length)

            tsize = images.shape[0]
            
            images_bank[index: index + tsize] = image_embeddings
            text_bank[index: index + tsize] = text_embeddings
            labels_bank[index:index + tsize] = labels
            index+=tsize

            batch_time.update(time.time() - timer)
            timer = time.time()
        
        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]

        #[t2i_top1, t2i_top10] = compute_topk(text_bank, images_bank, labels_bank, labels_bank, [1,10])
        #[i2t_top1, i2t_top10] = compute_topk(images_bank, text_bank, labels_bank, labels_bank, [1,10])

        i2t_top1, i2t_top5, i2t_top10, t2i_top1, t2i_top5, t2i_top10 = compute_topk(images_bank, text_bank, labels_bank, labels_bank, [1,5,10], True)
        i2t_mr, t2i_mr = compute_mr(images_bank, text_bank, labels_bank, labels_bank, 350, True)


        return i2t_top1, i2t_top5, i2t_top10, i2t_mr, t2i_top1, t2i_top5, t2i_top10, t2i_mr, batch_time.avg


def get_data_loader(image_dir, anno_dir, batch_size, split, max_length):

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_split = Fashion(image_dir, anno_dir, split, max_length, test_transform)

    loader = data.DataLoader(data_split, batch_size, shuffle=True, num_workers=2)

    return loader


def get_test_model_paths(ckpt_path):

    test_models = os.listdir(ckpt_path)
    test_models = list(filter(lambda x: os.path.isdir,test_models))
    test_models = sorted(test_models,key=lambda x: int(x.split(".")[0]))
    test_models = [os.path.join(ckpt_path,x) for x in test_models]

    return test_models


def get_network(args, model_path=None):

    network = Model(args)
    network = nn.DataParallel(network)
    # cudnn.benchmark = True
    args.start_epoch = 0

    # process network params
    if model_path==None:
        raise ValueError('Supply the model path with --model_path while testing')
    check_file(model_path, 'model_file')
    checkpoint = torch.load(model_path)
    args.start_epoch = checkpoint['epoch'] + 1
    network_dict = checkpoint['network']
    network.load_state_dict(network_dict) 
    print('==> Loading checkpoint "{}"'.format(model_path))

    return network



def main(args):
    # need to clear the pipeline
    # top1 & top10 need to be chosen in the same params ???

    test_loader = get_data_loader(args.image_dir, args.anno_dir, args.batch_size, 'test', args.max_length)

    i2t_top1 = 0.0
    i2t_top5 = 0.0
    i2t_top10 = 0.0
    i2t_mr = 0.0

    t2i_top1 = 0.0
    t2i_top5 = 0.0
    t2i_top10 = 0.0
    t2i_mr = 0.0

    test_models = get_test_model_paths(args.model_path)

    for model_path in test_models:

        network= get_network(args, model_path)

        i2t_top1_cur, i2t_top5_cur, i2t_top10_cur, i2t_mr_cur, t2i_top1_cur, t2i_top5_cur, t2i_top10_cur, t2i_mr_cur, test_time = get_metrics(test_loader, network, args)

        if t2i_top1_cur > t2i_top1:

            i2t_top1 = i2t_top1_cur
            i2t_top5 = i2t_top5_cur
            i2t_top10 = i2t_top10_cur

            t2i_top1 = t2i_top1_cur
            t2i_top5 = t2i_top5_cur
            t2i_top10 = t2i_top10_cur

            dst_best = os.path.join(args.model_path, 'model_best', 'best.pth.tar')
            shutil.copyfile(model_path, dst_best)


if __name__ == '__main__':
    
    args = get_test_args()
    main(args)