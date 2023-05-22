import sys
import os
import time
import argparse
from config import config
import util.misc as misc
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.utils import load_npy, build_transforms
import util.utils as utils
from monai.data import DataLoader, ImageDataset
from util.core import train_one_epoch
from models.resnet3D import get_cls_model
import json
import datetime

def get_args_parser():
    parser = argparse.ArgumentParser('Healthcare 3D-CNN_train', add_help=False)
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])

    parser = parser.parse_args()

    return parser

def update_config(config, args):

    config.defrost()
    config.ARCH = args.arch
    config.NAME = '3D_CNN'
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

    ##1. load the dataset
    train_list, _ = load_npy(config, npy=False)


    ##2. preprocess the images & build datasets
    transform_train = build_transforms(train=True)
    dataset_train = ImageDataset(image_files=train_list[0], labels=train_list[1], transform=transform_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    os.makedirs(config.TRAIN.OUTPUT_DIR, exist_ok=True)
    log_writer = SummaryWriter(log_dir=config.TRAIN.OUTPUT_DIR)
    data_loader_train = DataLoader(dataset_train, sampler=sampler_train,
                                                    batch_size=config.BATCH_PER_GPU,
                                                    num_workers=config.NUM_WORKERS,
                                                    pin_memory=config.PIN_MEM,
                                                    drop_last=True)


    ##3. build model
    model = get_cls_model(arch=config.ARCH,
                          n_classes=2,
                          n_input_channels=1,
                          shortcut_type='B',
                          conv1_t_size=7,
                          conv1_t_stride=1,
                          no_max_pool=False,
                          widen_factor=1.0)

    model.to(device)

    # scale learning rate
    if config.TRAIN.SCALE == 'linear':
        scaled_lr = config.TRAIN.LR * config.BATCH_PER_GPU / 256
    elif config.TRAIN.SCALE == 'const':
        scaled_lr = config.TRAIN.LR

    # set optimizer
    params_groups = utils.get_params_groups(model)
    if config.TRAIN.OPTIM == "adamw":
        optimizer = torch.optim.AdamW(params_groups, lr=scaled_lr)
    elif config.TRAIN.OPTIM == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=scaled_lr, momentum=0.9)


    # initialize scheduler
    lr_schedule = utils.cosine_scheduler(scaled_lr,
                                         config.TRAIN.MIN_LR,
                                         config.TRAIN.EPOCHS,
                                         len(data_loader_train),
                                         warmup_epochs=config.TRAIN.WARMUP,)

    wd_schedule = utils.cosine_scheduler(config.TRAIN.WEIGHT_DECAY,
                                         config.TRAIN.WEIGHT_DECAY_END,
                                         config.TRAIN.EPOCHS,
                                         len(data_loader_train),)

    criterion = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()



    ##4. train the model & save model
    print(f"Start training for {config.TRAIN.EPOCHS} epochs")
    start_time = time.time()
    for epoch in range(config.TRAIN.EPOCHS):
        if config.DDP.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model=model, data_loader=data_loader_train,
                                      optimizer=optimizer, criterion=criterion,
                                      lr_schedule=lr_schedule, wd_schedule=wd_schedule,
                                      device=device, epoch=epoch, loss_scaler=loss_scaler,
                                      log_writer=log_writer, config=config)


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}


        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(config.TRAIN.OUTPUT_DIR, f'{config.NAME}_log.txt'), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    misc.save_model(config=config.TRAIN, model=model,
                    optimizer=optimizer, loss_scaler=loss_scaler,name=config.NAME)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    main()
    print('Done!!!')
