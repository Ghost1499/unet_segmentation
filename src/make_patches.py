import cv2
import numpy as np
from scipy import ndimage
from image_proccess import refine

from my_fun_new_colab import kernel

# Определение функции для нарезания блоков размера wsize во входном изображении и создания паттернов
def createPtrns_Gray(img, wsize, ptrnTrain, hf):
    data = np.array(img, dtype="float32")
    data = cv2.GaussianBlur(data, (3, 3), 1)
    data = cv2.medianBlur(data, 5)
    if hf == "True":
        highpass_1 = ndimage.convolve(data, kernel)
        img = data * 0.75 + highpass_1
    elif hf == "Canny":
        img = cv2.Canny(data.astype("uint8"), 10, 100)
    width = img.shape[1]  # размеры входного изображения (по X)
    height = img.shape[0]  # размеры входного изображения (по Y)
    blocksW = int(width / wsize)  # число блоков
    blocksH = int(height / wsize)
    deltW = 0
    deltH = 0
    if width % wsize != 0:
        blocksW = blocksW + 1
        width_ = blocksW * wsize
        deltW = int((width_ - width) / blocksW)

    if height % wsize != 0:
        blocksH = blocksH + 1  #
        height_ = blocksH * wsize
        deltH = int((height_ - height) / blocksH)
    # step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
    for x in range(blocksW):  # *2):
        if x == 0:
            xmin = 0
            xmax = wsize
        elif x == 1:
            xmin = xmax - 2 * deltW
            xmax = xmin + wsize
        else:
            xmin = xmax - deltW
            xmax = xmin + wsize
        # print(xmin,xmax)
        for y in range(blocksH):  # *2):
            if y == 0:
                ymin = 0
                ymax = wsize
            elif y == 1:
                ymin = ymax - 2 * deltH
                ymax = ymin + wsize
            else:
                ymin = ymax - deltH
                ymax = ymin + wsize
            # print(ymin,ymax)

            imgRoi = img[ymin:ymax, xmin:xmax]
            imgRoi = imgRoi.astype("float32") / 255.0
            ptrnTrain.append(imgRoi)  # добавляет объект в конце списка


def createPtrns_Gray_AB(img_A, wsize, ptrnTrain_A, ptrnTrain_B, hf, por):
    if hf == "True":
        [img_A, img_B] = refine(img_A, por)

    elif hf == "Canny":
        img_B = cv2.Canny(img_A.astype("uint8"), 10, 100)
    width = img_A.shape[1]  # размеры входного изображения (по X)
    height = img_A.shape[0]  # размеры входного изображения (по Y)
    imgRoi_A = np.zeros((wsize, wsize), np.float32)
    imgRoi_B = np.zeros((wsize, wsize), np.float32)
    blocksW = int(width / wsize)  # число блоков
    blocksH = int(height / wsize)
    deltW = 0
    deltH = 0
    if width % wsize != 0:
        blocksW = blocksW + 1
        width_ = blocksW * wsize
        deltW = int((width_ - width) / blocksW)

    if height % wsize != 0:
        blocksH = blocksH + 1  #
        height_ = blocksH * wsize
        deltH = int((height_ - height) / blocksH)

    for x in range(blocksW):  # *2):
        if x == 0:
            xmin = 0
            xmax = wsize
        elif x == 1:
            xmin = xmax - 2 * deltW
            xmax = xmin + wsize
        else:
            xmin = xmax - deltW
            xmax = xmin + wsize
        # print(xmin,xmax)
        for y in range(blocksH):  # *2):
            if y == 0:
                ymin = 0
                ymax = wsize
            elif y == 1:
                ymin = ymax - 2 * deltH
                ymax = ymin + wsize
            else:
                ymin = ymax - deltH
                ymax = ymin + wsize
            # print(ymin,ymax)

            imgRoi_A = img_A[ymin:ymax, xmin:xmax]
            imgRoi_A = imgRoi_A.astype("float32") / 255.0
            imgRoi_B = img_B[ymin:ymax, xmin:xmax]
            imgRoi_B = imgRoi_B.astype("float32") / 255.0
            ptrnTrain_A.append(imgRoi_A)  # добавляет объект в конце списка
            ptrnTrain_B.append(imgRoi_B)


def createPtrns_Gray_2D(img, img_bin, wsize, ptrnTrain):
    data = np.array(img, dtype="float32")
    data = cv2.GaussianBlur(data, (5, 5), cv2.BORDER_DEFAULT)
    img = data + img_bin.astype("float32")
    width = img.shape[1]  # размеры входного изображения (по X)
    height = img.shape[0]  # размеры входного изображения (по Y)
    imgRoi = np.zeros((wsize, wsize), np.float32)

    blocksW = int(width / wsize)  # число блоков
    blocksH = int(height / wsize)
    deltW = 0
    deltH = 0
    if width % wsize != 0:
        blocksW = blocksW + 1
        width_ = blocksW * wsize
        deltW = int((width_ - width) / blocksW)

    if height % wsize != 0:
        blocksH = blocksH + 1  #
        height_ = blocksH * wsize
        deltH = int((height_ - height) / blocksH)

    for x in range(blocksW):  # *2):
        if x == 0:
            xmin = 0
            xmax = wsize
        elif x == 1:
            xmin = xmax - 2 * deltW
            xmax = xmin + wsize
        else:
            xmin = xmax - deltW
            xmax = xmin + wsize
        # print(xmin,xmax)
        for y in range(blocksH):  # *2):
            if y == 0:
                ymin = 0
                ymax = wsize
            elif y == 1:
                ymin = ymax - 2 * deltH
                ymax = ymin + wsize
            else:
                ymin = ymax - deltH
                ymax = ymin + wsize
            # print(ymin,ymax)

            imgRoi = img[ymin:ymax, xmin:xmax]
            # imgRoi[:,:,1] = img_hf[ymin : ymax, xmin : xmax]
            imgRoi = imgRoi / 255.0
            ptrnTrain.append(imgRoi)  # добавляет объект в конце списка


def createPtrns_Bin_AB(img_A, wsize, ptrnTrain_A, ptrnTrain_B):
    data = np.array(img_A, dtype="float32")
    # data = cv2.GaussianBlur(data,(3,3),1)
    # data = cv2.medianBlur(data, 5)
    data = ndimage.convolve(data, kernel)
    nmi = np.where(data < 128)
    data[nmi] = 0
    nma = np.where(data >= 128)
    data[nma] = 255
    img_B = cv2.dilate(
        data.astype("uint8"),
        cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2)),
        iterations=1,
    )
    img_B = img_B.astype("float32")
    width = img_A.shape[1]  # размеры входного изображения (по X)
    height = img_A.shape[0]  # размеры входного изображения (по Y)
    imgRoi_A = np.zeros((wsize, wsize), np.float32)
    imgRoi_B = np.zeros((wsize, wsize), np.float32)
    blocksW = int(width / wsize)  # число блоков
    blocksH = int(height / wsize)
    deltW = 0
    deltH = 0
    if width % wsize != 0:
        blocksW = blocksW + 1
        width_ = blocksW * wsize
        deltW = int((width_ - width) / blocksW)

    if height % wsize != 0:
        blocksH = blocksH + 1  #
        height_ = blocksH * wsize
        deltH = int((height_ - height) / blocksH)
    # step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
    for x in range(blocksW):  # *2):
        if x == 0:
            xmin = 0
            xmax = wsize
        elif x == 1:
            xmin = xmax - 2 * deltW
            xmax = xmin + wsize
        else:
            xmin = xmax - deltW
            xmax = xmin + wsize
        # print(xmin,xmax)
        for y in range(blocksH):  # *2):
            if y == 0:
                ymin = 0
                ymax = wsize
            elif y == 1:
                ymin = ymax - 2 * deltH
                ymax = ymin + wsize
            else:
                ymin = ymax - deltH
                ymax = ymin + wsize
            # print(ymin,ymax)

            imgRoi_A = img_A[ymin:ymax, xmin:xmax]
            imgRoi_A = imgRoi_A.astype("float32") / 255.0
            ptrnTrain_A.append(imgRoi_A)  # добавляет объект в
            imgRoi_B = img_B[ymin:ymax, xmin:xmax]
            imgRoi_B = imgRoi_B.astype("float32") / 255.0
            ptrnTrain_B.append(imgRoi_B)
            
            
def createPtrns_Bin(img, wsize, ptrnTrain, hf): # type: ignore
    if hf:
        data = np.array(img, dtype="float32")
        data = cv2.GaussianBlur(data, (3, 3), 1)
        data = cv2.medianBlur(data, 5)
        img = ndimage.convolve(data, kernel)
        nmi = np.where(img < 128)
        img[nmi] = 0
        nma = np.where(img >= 128)
        img[nma] = 255
    width = img.shape[1]  # размеры входного изображения (по X)
    height = img.shape[0]  # размеры входного изображения (по Y)
    blocksW = int(width / wsize)  # число блоков
    blocksH = int(height / wsize)
    deltW = 0
    deltH = 0
    if width % wsize != 0:
        blocksW = blocksW + 1
        width_ = blocksW * wsize
        deltW = int((width_ - width) / blocksW)

    if height % wsize != 0:
        blocksH = blocksH + 1  #
        height_ = blocksH * wsize
        deltH = int((height_ - height) / blocksH)
    # step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
    for x in range(blocksW):  # *2):
        if x == 0:
            xmin = 0
            xmax = wsize
        elif x == 1:
            xmin = xmax - 2 * deltW
            xmax = xmin + wsize
        else:
            xmin = xmax - deltW
            xmax = xmin + wsize
        # print(xmin,xmax)
        for y in range(blocksH):  # *2):
            if y == 0:
                ymin = 0
                ymax = wsize
            elif y == 1:
                ymin = ymax - 2 * deltH
                ymax = ymin + wsize
            else:
                ymin = ymax - deltH
                ymax = ymin + wsize
            # print(ymin,ymax)

            imgRoi = img[ymin:ymax, xmin:xmax]
            imgRoi = imgRoi.astype("float32") / 255.0
            ptrnTrain.append(imgRoi)  # добавляет объект в конце списка
            