import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from monai.transforms import (
    RandRotate90,
    Resize,
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
)

import pandas as pd

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

def build_transforms(train=True):

    if train:
        transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    else:
        transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    return transforms


def load_npy(config, npy=False):

    if npy:
        train_x = np.load('./X_train_voxel.npy')
        train_y = np.load('./Y_train_voxel.npy')
        test_x = np.load('./X_test_voxel.npy')
        test_y = np.load('./Y_test_voxel.npy')
        return (train_x, train_y), (test_x, test_y)

    else: # Split the dataset into train and test and save it as npy file.
        class_names = sorted(x for x in os.listdir(config.TRAIN.DATA_PATH))
        num_class = len(class_names)

        image_files = []
        labels = []
        for i in range(num_class):
            top_folder_path = os.path.join(config.TRAIN.DATA_PATH, class_names[i])
            for folder_path, _, file_names in os.walk(top_folder_path):
                for file_name in file_names:
                    file_path = os.path.join(folder_path, file_name)
                    image_files.append(file_path)
                    labels.append(i)


        train_x, test_x, train_y, test_y = train_test_split(image_files, labels,
                                                            test_size=0.1, shuffle=True,
                                                            stratify=labels,
                                                            random_state=config.SEED)

        np.save('./X_train_voxel', train_x)
        np.save('./Y_train_voxel', train_y)
        np.save('./X_test_voxel', test_x)
        np.save('./Y_test_voxel', test_y)

        return (train_x, train_y), (test_x, test_y)


def TADPOLE_load_npy(config, npy=False):

    if npy:
        train_x = np.load('./X_train_feature.npy')
        train_y = np.load('./Y_train_feature.npy')
        test_x = np.load('./X_test_feature.npy')
        test_y = np.load('./Y_test_feature.npy')
        return (train_x, train_y), (test_x, test_y)

    else: # Split the dataset into train and test and save it as npy file.

        pd.set_option('mode.chained_assignment', None)
        data = pd.read_csv('./data/TADPOLE_D1_D2.csv', sep=',', dtype='unicode')
        df = data[['RID', 'VISCODE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE', 'PTGENDER', 'PTEDUCAT', 'DX_bl']]
        df.drop_duplicates(subset='RID', keep='last', ignore_index=True, inplace=True)
        df = df.dropna()

        demo = df[['RID', 'VISCODE', 'AGE', 'PTGENDER', 'PTEDUCAT']]
        demo.to_csv('./demographic.csv', index=False)

        df = df[['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE', 'PTGENDER','PTEDUCAT', 'DX_bl']]
        df['PTGENDER'] = df['PTGENDER'].map({'Female': 1, 'Male': 0})
        df = df.replace('CN', 0)
        df.loc[df['DX_bl'] != 0, 'DX_bl'] = 1
        df = df.astype('float')

        x = df[['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE', 'PTGENDER', 'PTEDUCAT']].values
        y = df['DX_bl'].values

        train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                            test_size=0.1, shuffle=True,
                                                            stratify=y,
                                                            random_state=config.SEED)

        np.save('./X_train_feature', train_x)
        np.save('./Y_train_feature', train_y)
        np.save('./X_test_feature', test_x)
        np.save('./Y_test_feature', test_y)

        return (train_x, train_y), (test_x, test_y)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.labels = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.labels[index]












