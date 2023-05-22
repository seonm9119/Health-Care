import sys
import os
import argparse
from config import config
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from util.utils import load_npy, build_transforms
from util.core import evaluate
from models.resnet3D import get_cls_model
from monai.data import ImageDataset



def get_args_parser():
    parser = argparse.ArgumentParser('Healthcare 3D-CNN_test', add_help=False)
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])

    parser = parser.parse_args()

    return parser

def update_config(config, args):


    config.defrost()
    config.NAME = '3D_CNN'
    config.ARCH = args.arch
    config.freeze()


def main():
    ##0. initialize
    args = get_args_parser()
    update_config(config, args)

    if torch.cuda.is_available() is False:
        print('Does not support training without GPU.')
        sys.exit(1)

    device = torch.device(config.DEVICE)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    ##1. load the dataset (i.e, X_train.npy, Y_train.npy)
    train_list, test_list = load_npy(config, npy=True)


    ##2. preprocess the images & build datasets
    transform_test = build_transforms(train=False)
    dataset_train = ImageDataset(image_files=train_list[0], labels=train_list[1], transform=transform_test)
    dataset_test = ImageDataset(image_files=test_list[0], labels=test_list[1], transform=transform_test)


    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)


    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
                                                   batch_size=config.BATCH_PER_GPU,
                                                   num_workers=config.NUM_WORKERS,
                                                   pin_memory=config.PIN_MEM,
                                                   drop_last=False)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test,
                                                    batch_size=config.BATCH_PER_GPU,
                                                    num_workers=config.NUM_WORKERS,
                                                    pin_memory=config.PIN_MEM,
                                                    drop_last=False)

    ##3. build model
    model = get_cls_model(arch=config.ARCH,
                          n_classes=2,
                          n_input_channels=1)

    checkpoint = torch.load(f'{config.NAME}_checkpoint.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    train_stats = evaluate(config=config,
                           data_loader=data_loader_train,
                           model=model, device=device, header='Train Evaluate : ')

    test_stats = evaluate(config=config,
                           data_loader=data_loader_test,
                           model=model, device=device, header='Test Evaluate : ')



    os.makedirs(config.TRAIN.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.TRAIN.OUTPUT_DIR, "test_3D-CNN.txt"), mode="a", encoding="utf-8") as f:
        f.write(f'\n{train_stats}')
        f.write(f'\n{test_stats}')



if __name__ == '__main__':
    main()
    print('Done!!!')
