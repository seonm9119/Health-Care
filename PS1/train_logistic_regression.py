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
from util.utils import load_npy, build_transforms, MedNISTDataset
import json
import datetime
import math
from models.logistic import LogisticRegression
import util.utils as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Healthcare problem1_TRAIN', add_help=False)
    parser.add_argument('--data_path', default='D:/data/MedNIST', type=str, help='your data path')
    parser.add_argument('--arch', default='logistic', type=str)
    parser.add_argument('--output_dir', default='./save_logistic', type=str)

    parser = parser.parse_args()

    return parser

def update_config(config, args):

    config.defrost()
    config.DATA_PATH = args.data_path
    config.ARCH = args.arch
    config.freeze()


def train_one_epoch(model, data_loader, optimizer, criterion, lr_schedule,
                    wd_schedule, device, epoch, loss_scaler, log_writer=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (img, label) in enumerate(metric_logger.log_every(data_loader, config.PRINT_FREQ, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        it = len(data_loader) * epoch + data_iter_step  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        _, _, h, w = img.shape
        img = img.view(-1, h * w)

        img = img.to(device, non_blocking=True)
        label = label.to(device)

        with torch.cuda.amp.autocast():
            pred = model(img)
            loss = criterion(pred, label)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)

        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)


        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def main():

    args = get_args_parser()
    update_config(config, args)

    if torch.cuda.is_available() is False:
        print('Does not support training without GPU.')
        sys.exit(1)

    device = torch.device(config.DEVICE)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    ##1. load the MedMNIST dataset (i.e, X_train.npy, Y_train.npy)
    train_list, _ = load_npy(config, npy=True)

    ##2. preprocess the images & build datasets
    transform_train = build_transforms(config, train=True)

    dataset_train = MedNISTDataset(train_list, config.DATA_PATH, transform=transform_train)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    os.makedirs(config.LOGISTIC.OUTPUT_DIR, exist_ok=True)
    log_writer = SummaryWriter(log_dir=config.LOGISTIC.OUTPUT_DIR)

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
                                                    batch_size=config.BATCH_PER_GPU,
                                                    num_workers=config.NUM_WORKERS,
                                                    pin_memory=config.PIN_MEM)

    ##3. build model
    model = LogisticRegression(config.IMG_SIZE*config.IMG_SIZE, num_classes=6)
    model.to(device)


    if config.LOGISTIC.SCALE == 'linear':
        scaled_lr = config.LOGISTIC.LR * config.BATCH_PER_GPU / 256
    elif config.LOGISTIC.SCALE == 'const':
        scaled_lr = config.LOGISTIC.LR

    # Set optimizer
    params_groups = utils.get_params_groups(model)
    if config.LOGISTIC.OPTIM == "adamw":
        optimizer = torch.optim.AdamW(params_groups, lr=scaled_lr)  # to use with ViTs
    elif config.LOGISTIC.OPTIM == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=scaled_lr, momentum=0.9)  # lr is set by scheduler


    lr_schedule = utils.cosine_scheduler(scaled_lr,
                                         config.LOGISTIC.MIN_LR,
                                         config.LOGISTIC.EPOCHS,
                                         len(data_loader_train),
                                         warmup_epochs=config.LOGISTIC.WARMUP,)

    wd_schedule = utils.cosine_scheduler(config.TRAIN.WEIGHT_DECAY,
                                         config.TRAIN.WEIGHT_DECAY_END,
                                         config.LOGISTIC.EPOCHS,
                                         len(data_loader_train),)



    criterion = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()

    ##4. train the model & save model
    print(f"Start training for {config.LOGISTIC.EPOCHS} epochs")
    start_time = time.time()
    for epoch in range(config.LOGISTIC.EPOCHS):

        train_stats = train_one_epoch(model=model, data_loader=data_loader_train, optimizer=optimizer,
                                      criterion=criterion, lr_schedule=lr_schedule, wd_schedule=wd_schedule,
                                      device=device, epoch=epoch, loss_scaler=loss_scaler,
                                      log_writer=log_writer)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}


        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(config.LOGISTIC.OUTPUT_DIR, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    misc.save_model(config=config.LOGISTIC, model=model,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    main()
    print("Done!!!")