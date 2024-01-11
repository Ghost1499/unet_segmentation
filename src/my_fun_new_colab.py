# Импорт всех необходимых компонентов и утилит
# Установка библиотеки TensorFlow-Advanced-Segmentation-Models с разными функциями ошибок и сегментационными моделями


import cv2  # это opencv
import numpy as np
from keras import backend as K  # Для программирования собственных метрик
from tensorflow.keras.utils import plot_model
from keras.layers import Dropout
from keras.models import Sequential, load_model
from keras import optimizers
from keras.metrics import Precision, Recall
from scipy import ndimage

from metrics import F1


clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(32, 32))  # адаптивная эквализация
# tf.config.run_functions_eagerly(True)
kernel = np.array(
    [
        [-1, -1, -1, -1, -1],
        [-1, 1, 2, 1, -1],
        [-1, 2, 4, 2, -1],
        [-1, 1, 2, 1, -1],
        [-1, -1, -1, -1, -1],
    ]
)

def shuffle_unson_write_AB(
    basePath, end_path, ptrnsTrainX_A, ptrnsTrainX_B, ptrnsTrainY_A, ptrnsTrainY_B
):
    # Перемешивание данных для последдующей записи
    a = ptrnsTrainX_A.shape
    b = ptrnsTrainY_A.shape
    assert a[0] == b[0]
    np.random.seed(10)
    p = np.random.permutation(a[0])

    ptrnsTrainX_A_ = ptrnsTrainX_A[p, :, :, :]
    ptrnsTrainX_B_ = ptrnsTrainX_B[p, :, :, :]
    ptrnsTrainY_A_ = ptrnsTrainY_A[p, :, :, :]
    ptrnsTrainY_B_ = ptrnsTrainY_B[p, :, :, :]
    # end_path='_single_'+str(wsize)+'_'+str(out_size)+'_'+str(Ap_norm)+'_'+str(Ap_clahe)+'_'+str(por)
    if basePath == "train_colab":
        np.save(
            basePath + "/gray/ptrnsTrainX_AB" + end_path + ".npy",
            (ptrnsTrainX_A_, ptrnsTrainX_B_),
        )
        np.save(
            basePath + "/binary/ptrnsTrainY_AB" + end_path + ".npy",
            (ptrnsTrainY_A_, ptrnsTrainY_B_),
        )
    elif basePath == "val_colab":
        np.save(
            basePath + "/gray/ptrnsValX_AB" + end_path + ".npy",
            (ptrnsTrainX_A_, ptrnsTrainX_B_),
        )
        # ptrnsTrainX=np.load(basePath+'/gray/ptrnsTrainX.npy');#проверка
        np.save(
            basePath + "/binary/ptrnsValY_AB" + end_path + ".npy",
            (ptrnsTrainY_A_, ptrnsTrainY_B_),
        )

def shuffle_unson_write(
    basePath, ptrnsTrainX, ptrnsTrainY, wsize, out_size, Ap_norm, Ap_clahe
):
    # Перемешивание данных для последдующей записи

    a = ptrnsTrainX.shape
    b = ptrnsTrainY.shape
    assert a[0] == b[0]
    np.random.seed(10)
    p = np.random.permutation(a[0])

    ptrnsTrainX_ = ptrnsTrainX[p, :, :, :]
    ptrnsTrainY_ = ptrnsTrainY[p, :, :, :]
    end_path = (
        "_single_"
        + str(wsize)
        + "_"
        + str(out_size)
        + "_"
        + str(Ap_norm)
        + "_"
        + str(Ap_clahe)
    )
    if basePath == "train_colab":
        np.save(basePath + "/gray/ptrnsTrainX" + end_path + ".npy", ptrnsTrainX_)
        # ptrnsTrainX=np.load(basePath+'/gray/ptrnsTrainX.npy');#проверка
        np.save(basePath + "/binary/ptrnsTrainY" + end_path + ".npy", ptrnsTrainY_)
    elif basePath == "val_colab":
        np.save(basePath + "/gray/ptrnsValX" + end_path + ".npy", ptrnsTrainX_)
        # ptrnsTrainX=np.load(basePath+'/gray/ptrnsTrainX.npy');#проверка
        np.save(basePath + "/binary/ptrnsValY" + end_path + ".npy", ptrnsTrainY_)

    # ptrnsTrainY=np.load(basePath+'/binary'/ptrnsTrainY.npy);#проверк
    # return ptrnsTrainX_,ptrnsTrainY_



