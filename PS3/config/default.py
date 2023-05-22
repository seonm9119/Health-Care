from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yacs.config import CfgNode as CN

# Initialize parameters
_C = CN()
_C.SEED = 0
_C.BATCH_PER_GPU = 16
_C.DEVICE = 'cuda'
_C.NUM_WORKERS = 2
_C.PIN_MEM = True
_C.PRINT_FREQ = 5
_C.ARCH = ''
_C.IMG_SIZE = 96
_C.NAME = ''

# for CNN
_C.TRAIN = CN()
_C.TRAIN.DATA_PATH = '.\\data\\ADNI'
_C.TRAIN.OUTPUT_DIR = './'
_C.TRAIN.SCALE = 'const'
_C.TRAIN.LR = 1e-4
_C.TRAIN.MIN_LR = 0
_C.TRAIN.EPOCHS = 20
_C.TRAIN.WARMUP = 0
_C.TRAIN.OPTIM = 'adamw'
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.WEIGHT_DECAY_END = 0.5


# Distributed training parameters
_C.DDP = CN()
_C.DDP.DISTRIBUTED = False



if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)