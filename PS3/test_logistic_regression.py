from config import config
import torch
import numpy as np
from util.utils import TADPOLE_load_npy
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import roc_auc_score



def main():

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    ##1. load the MedMNIST dataset (i.e, X_train.npy, Y_train.npy)
    (train_x, train_y), (test_x, test_y) = TADPOLE_load_npy(config, npy=True)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)

    model = joblib.load('model.pkl')
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)


    with open('test_logistic-regression.txt', mode="a", encoding="utf-8") as f:
        f.write(f'\n{roc_auc_score(train_y, train_pred)}')
        f.write(f'\n{roc_auc_score(test_y, test_pred)}')

    return



if __name__ == '__main__':
    main()
    print("Done!!!")
