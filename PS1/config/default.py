from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yacs.config import CfgNode as CN

# Initialize parameters
_C = CN()
_C.SEED = 0
_C.DATA_PATH = ''
_C.BATCH_PER_GPU = 32
_C.DEVICE = 'cuda'
_C.NUM_WORKERS = 4
_C.PIN_MEM = True
_C.PRINT_FREQ = 50
_C.ARCH = ''
_C.IMG_SIZE = 64

# for CNN
_C.TRAIN = CN()
_C.TRAIN.OUTPUT_DIR = './save_CNN'
_C.TRAIN.CHECKPOINT = './save_CNN/checkpoint.pth'
_C.TRAIN.SCALE = 'linear'
_C.TRAIN.LR = 1e-3
_C.TRAIN.MIN_LR = 0.
_C.TRAIN.EPOCHS = 5
_C.TRAIN.WARMUP = 0
_C.TRAIN.OPTIM = 'adamw'
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.WEIGHT_DECAY_END = 0.5

# for logistic regression
_C.LOGISTIC = CN()
_C.LOGISTIC.ARCH = 'logistic'
_C.LOGISTIC.OUTPUT_DIR = './save_logistic'
_C.LOGISTIC.CHECKPOINT = './save_logistic/checkpoint.pth'
_C.LOGISTIC.SCALE = 'const'
_C.LOGISTIC.LR = 0.003
_C.LOGISTIC.MIN_LR = 0.
_C.LOGISTIC.EPOCHS = 10
_C.LOGISTIC.WARMUP = 0
_C.LOGISTIC.OPTIM = 'adamw'

# Data augmentation
_C.AUG = CN()
_C.AUG.PROB = 0.5
_C.AUG.MIN_ZOOM = 0.9
_C.AUG.MAX_ZOOM = 1.1
_C.AUG.SPATIAL_AXIS = 0
_C.AUG.RANGE_X = 12
_C.AUG.KEEP_SIZE = True

# Distributed training parameters
_C.DDP = CN()
_C.DDP.DISTRIBUTED = False



if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)