import sys
import os
import argparse
from config import config
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from util.utils import load_npy, build_transforms, MedNISTDataset
from util.core import evaluate
from models.build_models import get_cls_model


def get_args_parser():
    parser = argparse.ArgumentParser('Healthcare problem1_TEST', add_help=False)
    parser.add_argument('--data_path', default='D:/data/MedNIST', type=str, help='your data path')
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                                 'vit_tiny', 'vit_small', 'vit_base', 'vit_large'])

    parser = parser.parse_args()

    return parser

def update_config(config, args):


    config.defrost()
    config.DATA_PATH = args.data_path
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

    ##1. load the MedMNIST dataset (i.e, X_train.npy, Y_train.npy)
    train_list, test_list = load_npy(config, npy=True)

    ##2. preprocess the images & build datasets
    transform_test = build_transforms(config, train=False)

    dataset_train = MedNISTDataset(train_list, config.DATA_PATH, transform=transform_test)
    dataset_test = MedNISTDataset(test_list, config.DATA_PATH, transform=transform_test)


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
    model = get_cls_model(config, num_classes=6)

    checkpoint = torch.load(config.TRAIN.CHECKPOINT, map_location='cpu')
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
    with open(os.path.join(config.TRAIN.OUTPUT_DIR, "20214029-2D-CNN-regression.txt"), mode="a", encoding="utf-8") as f:
        f.write('20214029')
        f.write('\n*** Train accuracy {0}\n{1}'.format('*'*40, train_stats))
        f.write('\n*** Test accuracy {0}\n{1}'.format('*'*40, test_stats))


if __name__ == '__main__':
    main()
    print('Done!!!')
