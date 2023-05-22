from config import config
import numpy as np
from util.utils import TADPOLE_load_npy
from sklearn.preprocessing import StandardScaler
import joblib


def main():


    np.random.seed(config.SEED)

    (train_x, train_y), _ = TADPOLE_load_npy(config, npy=False)

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)

    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10,), solver='sgd', learning_rate_init=0.1, max_iter=1000)

    mlp.fit(train_x, train_y)
    joblib.dump(mlp, 'model.pkl')

    return





if __name__ == '__main__':
    main()
    print("Done!!!")