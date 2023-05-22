import math
import sys
import torch
import util.misc as misc
from sklearn.metrics import classification_report


def train_one_epoch(model, data_loader, optimizer,
                    criterion, lr_schedule, wd_schedule, device,  epoch, loss_scaler,
                    log_writer=None, config=None):


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


        img = img.to(device, non_blocking=True)
        label = label.to(device)


        with torch.cuda.amp.autocast():
            pred = model(img)
            loss = criterion(pred, label)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=True)

        optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)


        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(config, data_loader, model, device, header):

    metric_logger = misc.MetricLogger(delimiter="  ")
    model.eval()

    y_true = []
    y_pred = []
    for (images, labels) in metric_logger.log_every(data_loader, config.PRINT_FREQ, header):

        if config.ARCH == 'logistic':
            _, _, h, w = images.shape
            images = images.view(-1, h * w)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device)


        with torch.no_grad():
            pred = model(images).argmax(dim=1)

            for i in range(len(pred)):
                y_true.append(labels[i].item())
                y_pred.append(pred[i].item())

    class_names = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
    return classification_report(y_true, y_pred, target_names=class_names, digits=4)



