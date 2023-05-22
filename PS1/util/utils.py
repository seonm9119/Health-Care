import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def build_transforms(config, train=True):

    cfg = config.AUG
    if train:
        transforms = Compose([LoadImage(image_only=True),
                              EnsureChannelFirst(),
                              ScaleIntensity(),
                              RandRotate(range_x=np.pi / cfg.RANGE_X, prob=cfg.PROB, keep_size=cfg.KEEP_SIZE),
                              RandFlip(spatial_axis=cfg.SPATIAL_AXIS, prob=cfg.PROB),
                              RandZoom(min_zoom=cfg.MIN_ZOOM, max_zoom=cfg.MAX_ZOOM, prob=cfg.PROB),
                              ])
    else:
        transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

    return transforms


def load_npy(config, npy=False):

    if npy:
        train_x = np.load('./npy/X_train.npy')
        train_y = np.load('./npy/Y_train.npy')
        test_x = np.load('./npy/X_test.npy')
        test_y = np.load('./npy/Y_test.npy')
        return (train_x, train_y), (test_x, test_y)

    else: # Split the dataset into train and test and save it as npy file.
        class_names = sorted(x for x in os.listdir(config.DATA_PATH))
        num_class = len(class_names)

        image_files = []
        labels = []
        for i in range(num_class):
            for x in os.listdir(os.path.join(config.DATA_PATH, class_names[i])):
                image_files.append(f'{class_names[i]}/{x}')
                labels.append(i)

        train_x, test_x, train_y, test_y = train_test_split(image_files, labels,
                                                            test_size=0.1, shuffle=True,
                                                            stratify=labels,
                                                            random_state=config.SEED)


        np.save('./npy/X_train', train_x)
        np.save('./npy/Y_train', train_y)
        np.save('./npy/X_test', test_x)
        np.save('./npy/Y_test', test_y)

        return (train_x, train_y), (test_x, test_y)


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_path, transform=None):
        self.image_files, labels = data
        self.data_path = data_path
        self.labels = torch.LongTensor(labels)
        self.transforms = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        return self.transforms(os.path.join(self.data_path, self.image_files[index])), self.labels[index]












