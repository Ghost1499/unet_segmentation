import os

import cv2
import numpy as np

from image_proccess import pred_norm_img
from make_patches import createPtrns_Bin_AB, createPtrns_Gray, createPtrns_Gray_AB,createPtrns_Bin
from my_fun_new_colab import clahe


def loadTrainData_binary_colab_AB(basePath, wsize, Ap_norm, Ap_clahe, hf, por):
    """
    Опредление функции для загрузки данных из папки с заданным путем basePath и формирования обучающей выборки с нарезкой
    """
    # out_size==2
    ptrnsTrainX_A = []
    ptrnsTrainX_B = []
    ptrnsTrainY_A = []
    ptrnsTrainY_B = []
    # смешанные данные
    # path_gray = basePath + "/gray_mixt"
    # path_binary = basePath + "/binary_mixt"
    # данные из одного слоя
    # path_gray = basePath +'/gray 2014/1me'
    path_gray = basePath + "/gray 2014/1me"
    path_binary = basePath + "/binary 2014/1me"
    k = 0
    l = 0
    # os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри
    for fileName in os.listdir(path_gray):  # дает все что есть в каталоге
        k = k + 1
        if k % 5 == 0:  # остаток от деления
            l = l + 1
            imgGray = cv2.imread(path_gray + "/" + fileName, 0)
            width = imgGray.shape[1]  # размеры входного изображения (по X)
            height = imgGray.shape[0]

            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((height,width))
            # imgGray=((img_+ss)*255).astype('uint8')

            if Ap_clahe == True and Ap_norm == False:
                imgGray = clahe.apply(imgGray)
            if Ap_clahe == False and Ap_norm == True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm == True and Ap_clahe == True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print("name ", fileName)

            createPtrns_Gray_AB(imgGray, wsize, ptrnsTrainX_A, ptrnsTrainX_B, hf, por)
            imgGray_ = imgGray.T
            createPtrns_Gray_AB(imgGray_, wsize, ptrnsTrainX_A, ptrnsTrainX_B, hf, por)
    print(l)

    k = 0
    l = 0
    for fileName in os.listdir(path_binary):
        k = k + 1
        if k % 5 == 0:
            l = l + 1
            imgBin = cv2.imread(path_binary + "/" + fileName, 0)
            print("name ", fileName)
            createPtrns_Bin_AB(imgBin, wsize, ptrnsTrainY_A, ptrnsTrainY_B)
            imgBin_ = imgBin.T
            createPtrns_Bin_AB(imgBin_, wsize, ptrnsTrainY_A, ptrnsTrainY_B)
    print(l)

    ptrnsTrainX_A = np.array(ptrnsTrainX_A)
    ptrnsTrainX_A = ptrnsTrainX_A.reshape(ptrnsTrainX_A.shape[0], wsize, wsize, 1)  # 3
    ptrnsTrainX_B = np.array(ptrnsTrainX_B)
    ptrnsTrainX_B = ptrnsTrainX_B.reshape(ptrnsTrainX_B.shape[0], wsize, wsize, 1)  # 3

    # развертка в матрицу изображения
    ptrnsTrainY_A = np.array(ptrnsTrainY_A)
    ptrnsTrainY_A = ptrnsTrainY_A.reshape(ptrnsTrainY_A.shape[0], wsize, wsize, 1)  # 3
    ptrnsTrainY_B = np.array(ptrnsTrainY_B)
    ptrnsTrainY_B = ptrnsTrainY_B.reshape(ptrnsTrainY_B.shape[0], wsize, wsize, 1)  # 3
    return (ptrnsTrainX_A, ptrnsTrainX_B, ptrnsTrainY_A, ptrnsTrainY_B)


def loadValData_binary_colab_AB(basePath, wsize, Ap_norm, Ap_clahe, hf, por):
    """
    Создание выборки для валидации
    """
    ptrnsValX_A = []
    ptrnsValX_B = []
    ptrnsValY_A = []
    ptrnsValY_B = []
    # смешанные данные
    # path_gray = basePath + "/gray_mixt"
    # path_binary = basePath + "/binary_mixt"
    # данные из одного слоя
    # path_gray = basePath +'/gray 2014/1me'
    path_gray = basePath + "/gray 2014/1me"
    path_binary = basePath + "/binary 2014/1me"
    k = 0
    l = 0
    # os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри
    for fileName in os.listdir(path_gray):  # дает все что есть в каталоге
        k = k + 1
        if (k - 1) % 10 == 0:
            l = l + 1
            imgGray = cv2.imread(path_gray + "/" + fileName, 0)
            width = imgGray.shape[1]  # размеры входного изображения (по X)
            height = imgGray.shape[0]

            if Ap_clahe == True and Ap_norm == False:
                imgGray = clahe.apply(imgGray)
            if Ap_clahe == False and Ap_norm == True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm == True and Ap_clahe == True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print("name ", fileName)

            createPtrns_Gray_AB(imgGray, wsize, ptrnsValX_A, ptrnsValX_B, hf, por)
            imgGray_ = imgGray.T
            createPtrns_Gray_AB(imgGray_, wsize, ptrnsValX_A, ptrnsValX_B, hf, por)
    print(l)

    k = 0
    l = 0
    for fileName in os.listdir(path_binary):
        k = k + 1
        if (k - 1) % 10 == 0:
            l = l + 1
            imgBin = cv2.imread(path_binary + "/" + fileName, 0)
            print("name ", fileName)
            createPtrns_Bin_AB(imgBin, wsize, ptrnsValY_A, ptrnsValY_B)
            imgBin_ = imgBin.T
            createPtrns_Bin_AB(imgBin_, wsize, ptrnsValY_A, ptrnsValY_B)
    print(l)
    ptrnsValX_A = np.array(ptrnsValX_A)
    ptrnsValX_A = ptrnsValX_A.reshape(ptrnsValX_A.shape[0], wsize, wsize, 1)  # 3
    ptrnsValX_B = np.array(ptrnsValX_B)
    ptrnsValX_B = ptrnsValX_B.reshape(ptrnsValX_B.shape[0], wsize, wsize, 1)  # 3
    ptrnsValY_A = np.array(ptrnsValY_A)
    ptrnsValY_A = ptrnsValY_A.reshape(ptrnsValX_A.shape[0], wsize, wsize, 1)  # 3
    ptrnsValY_B = np.array(ptrnsValY_B)
    ptrnsValY_B = ptrnsValY_B.reshape(ptrnsValX_B.shape[0], wsize, wsize, 1)  # 3
    return (ptrnsValX_A, ptrnsValX_B, ptrnsValY_A, ptrnsValY_B)


def loadTestData_binary_colab_AB(basePath, wsize, Ap_norm, Ap_clahe, hf, por):
    # out_size==2
    ptrnsTestX_A = []
    ptrnsTestX_B = []
    ptrnsTestY_A = []
    ptrnsTestY_B = []
    path_gray = basePath + "/gray 2014/1me"

    k = 0
    # os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри
    for fileName in os.listdir(path_gray):  # дает все что есть в каталоге
        k = k + 1
        if k % 5 != 0:  # остаток от деления
            imgGray = cv2.imread(path_gray + "/" + fileName, 0)

            if Ap_clahe == True:
                imgGray = clahe.apply(imgGray)
            if Ap_norm == True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm == True and Ap_clahe == True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print("name ", fileName)
            createPtrns_Gray_AB(imgGray, wsize, ptrnsTestX_A, ptrnsTestX_B, hf, por)
            imgGray_ = imgGray.T
            createPtrns_Gray_AB(imgGray_, wsize, ptrnsTestX_A, ptrnsTestX_B, hf, por)
    path = basePath + "/binary 2014/1me"
    k = 0
    for fileName in os.listdir(path):
        k = k + 1
        if k % 5 != 0:
            imgBin = cv2.imread(path + "/" + fileName, 0)
            print("name ", fileName)
            createPtrns_Bin_AB(imgBin, wsize, ptrnsTestY_A, ptrnsTestY_B)
            imgBin_ = imgBin.T
            createPtrns_Bin_AB(imgBin_, wsize, ptrnsTestY_A, ptrnsTestY_B)

    ptrnsTestX_A = np.array(ptrnsTestX_A)
    ptrnsTestX_A = ptrnsTestX_A.reshape(ptrnsTestX_A.shape[0], wsize, wsize, 1)  # 3
    ptrnsTestX_B = np.array(ptrnsTestX_B)
    ptrnsTestX_B = ptrnsTestX_B.reshape(ptrnsTestX_B.shape[0], wsize, wsize, 1)  # 3

    # развертка в матрицу изображения
    ptrnsTestY_A = np.array(ptrnsTestY_A)
    ptrnsTestY_A = ptrnsTestY_A.reshape(ptrnsTestY_A.shape[0], wsize, wsize, 1)  # 3
    ptrnsTestY_B = np.array(ptrnsTestY_B)
    ptrnsTestY_B = ptrnsTestY_B.reshape(ptrnsTestY_B.shape[0], wsize, wsize, 1)  # 3
    return (ptrnsTestX_A, ptrnsTestX_B, ptrnsTestY_A, ptrnsTestY_B)


def loadTestData_binary_colab(basePath, wsize, Ap_norm, Ap_clahe, hf):
    """
    Опредление функции для загрузки данных из папки с заданным путем basePath и формирования тестирующей выборки с нарезкой
    """
    # out_size==2
    ptrnsTestX = []
    ptrnsTestY = []

    path_gray = basePath + "/gray 2014/1me"
    path_BIN = basePath + "/gray 2014/train_colab_bin"

    k = 0
    # os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри
    for fileName in os.listdir(path_gray):  # дает все что есть в каталоге
        k = k + 1
        if k % 5 != 0:  # остаток от деления
            imgGray = cv2.imread(path_gray + "/" + fileName, 0)
            fileName_ = fileName[0 : len(fileName) - 4] + "_3.jpg"
            imgBIN = cv2.imread(path_BIN + "/" + fileName_, 0)
            # width = imgGray.shape[1]
            # height = imgGray.shape[0]
            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((2000,2000))
            # imgGray=((img_+ss)*255).astype('uint8')

            if Ap_clahe == True:
                imgGray = clahe.apply(imgGray)
            if Ap_norm == True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm == True and Ap_clahe == True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print("name ", fileName)
            print("name ", fileName_)

            # createPtrns_Gray_2D(imgGray,imgBIN, wsize, ptrnsTestX)
            # imgGray_=imgGray.T
            # imgBIN_=imgBIN.T
            # createPtrns_Gray_2D(imgGray_,imgBIN_, wsize, ptrnsTestX)
            createPtrns_Gray(imgGray, wsize, ptrnsTestX, hf)
            imgGray_ = imgGray.T
            createPtrns_Gray(imgGray_, wsize, ptrnsTestX, hf)
    path = basePath + "/binary 2014/1me"
    k = 0
    for fileName in os.listdir(path):
        k = k + 1
        if k % 5 != 0:
            imgBin = cv2.imread(path + "/" + fileName, 0)
            print("name ", fileName)
            createPtrns_Bin(imgBin, wsize, ptrnsTestY, False)
            imgBin_ = imgBin.T
            createPtrns_Bin(imgBin_, wsize, ptrnsTestY, False)

    ptrnsTestX = np.array(ptrnsTestX)
    # развертка в матрицу изображения
    ptrnsTestX = ptrnsTestX.reshape(ptrnsTestX.shape[0], wsize, wsize, 1)  # 3

    # развертка в матрицу изображения
    ptrnsTestY = np.array(ptrnsTestY)
    ptrnsTestY = ptrnsTestY.reshape(ptrnsTestY.shape[0], wsize, wsize, 1)  # 3
    return (ptrnsTestX, ptrnsTestY)


def loadTrainData_binary_colab(basePath, wsize, Ap_norm, Ap_clahe, hf):
    """
    Опредление функции для загрузки данных из папки с заданным путем basePath и формирования обучающей выборки с нарезкой
    """
    # out_size==2
    ptrnsTrainX = []
    ptrnsTrainY = []
    # смешанные данные
    # path_gray = basePath + "/gray_mixt"
    # path_binary = basePath + "/binary_mixt"
    # данные из одного слоя
    # path_gray = basePath +'/gray 2014/1me'
    path_gray = basePath + "/gray 2014/1me"
    path_BIN = basePath + "/gray 2014/train_colab_bin"
    path_binary = basePath + "/binary 2014/1me"
    k = 0
    l = 0
    # os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри
    for fileName in os.listdir(path_gray):  # дает все что есть в каталоге
        k = k + 1
        if k % 5 == 0:  # остаток от деления
            l = l + 1
            imgGray = cv2.imread(path_gray + "/" + fileName, 0)
            fileName_ = fileName[0 : len(fileName) - 4] + "_3.jpg"
            imgBIN = cv2.imread(path_BIN + "/" + fileName_, 0)
            width = imgGray.shape[1]  # размеры входного изображения (по X)
            height = imgGray.shape[0]

            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((height,width))
            # imgGray=((img_+ss)*255).astype('uint8')

            if Ap_clahe == True and Ap_norm == False:
                imgGray = clahe.apply(imgGray)
            if Ap_clahe == False and Ap_norm == True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm == True and Ap_clahe == True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print("name ", fileName)
            print("name ", fileName_)

            # createPtrns_Gray_2D(imgGray,imgBIN, wsize, ptrnsTrainX)
            # imgGray_=imgGray.T
            # imgBIN_=imgBIN.T
            # createPtrns_Gray_2D(imgGray_,imgBIN_,wsize, ptrnsTrainX)

            createPtrns_Gray(imgGray, wsize, ptrnsTrainX, hf)
            imgGray_ = imgGray.T
            createPtrns_Gray(imgGray_, wsize, ptrnsTrainX, hf)
    print(l)

    k = 0
    l = 0
    for fileName in os.listdir(path_binary):
        k = k + 1
        if k % 5 == 0:
            l = l + 1
            imgBin = cv2.imread(path_binary + "/" + fileName, 0)
            print("name ", fileName)
            createPtrns_Bin(imgBin, wsize, ptrnsTrainY, False)
            imgBin_ = imgBin.T
            createPtrns_Bin(imgBin_, wsize, ptrnsTrainY, False)
    print(l)

    ptrnsTrainX = np.array(ptrnsTrainX)
    # развертка в матрицу изображения
    ptrnsTrainX = ptrnsTrainX.reshape(ptrnsTrainX.shape[0], wsize, wsize, 1)  # 3

    # развертка в матрицу изображения
    ptrnsTrainY = np.array(ptrnsTrainY)
    ptrnsTrainY = ptrnsTrainY.reshape(ptrnsTrainY.shape[0], wsize, wsize, 1)  # 3
    return (ptrnsTrainX, ptrnsTrainY)


def loadValData_binary_colab(basePath, wsize, Ap_norm, Ap_clahe, hf):
    """
    Создание выборки для валидации
    """
    ptrnsValX = []
    ptrnsValY = []
    # смешанные данные
    # path_gray = basePath + "/gray_mixt"
    # path_binary = basePath + "/binary_mixt"
    # данные из одного слоя
    # path_gray = basePath +'/gray 2014/1me'
    path_gray = basePath + "/gray 2014/1me"
    path_BIN = basePath + "/gray 2014/train_colab_bin"
    path_binary = basePath + "/binary 2014/1me"
    k = 0
    l = 0
    # os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри
    for fileName in os.listdir(path_gray):  # дает все что есть в каталоге
        k = k + 1
        if (k - 1) % 10 == 0:
            l = l + 1
            imgGray = cv2.imread(path_gray + "/" + fileName, 0)
            fileName_ = fileName[0 : len(fileName) - 4] + "_3.jpg"
            imgBIN = cv2.imread(path_BIN + "/" + fileName_, 0)
            width = imgGray.shape[1]  # размеры входного изображения (по X)
            height = imgGray.shape[0]
            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((height,width))
            # imgGray=((img_+ss)*255).astype('uint8')

            if Ap_clahe == True and Ap_norm == False:
                imgGray = clahe.apply(imgGray)
            if Ap_clahe == False and Ap_norm == True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm == True and Ap_clahe == True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print("name ", fileName)
            print("name ", fileName_)

            # createPtrns_Gray_2D(imgGray,imgBIN, wsize, ptrnsValX)
            # imgGray_=imgGray.T
            # imgBIN_=imgBIN.T
            # createPtrns_Gray_2D(imgGray_,imgBIN_, wsize, ptrnsValX)

            createPtrns_Gray(imgGray, wsize, ptrnsValX, hf)
            imgGray_ = imgGray.T
            createPtrns_Gray(imgGray_, wsize, ptrnsValX, hf)
    print(l)

    k = 0
    l = 0
    for fileName in os.listdir(path_binary):
        k = k + 1
        if (k - 1) % 10 == 0:
            l = l + 1
            imgBin = cv2.imread(path_binary + "/" + fileName, 0)
            print("name ", fileName)
            createPtrns_Bin(imgBin, wsize, ptrnsValY, False)
            imgBin_ = imgBin.T
            createPtrns_Bin(imgBin_, wsize, ptrnsValY, False)
    print(l)
    ptrnsValX = np.array(ptrnsValX)
    # развертка в матрицу изображения
    ptrnsValX = ptrnsValX.reshape(ptrnsValX.shape[0], wsize, wsize, 1)  # 3
    # развертка в матрицу изображения
    ptrnsValY = np.array(ptrnsValY)
    ptrnsValY = ptrnsValY.reshape(ptrnsValY.shape[0], wsize, wsize, 1)  # 3
    return (ptrnsValX, ptrnsValY)