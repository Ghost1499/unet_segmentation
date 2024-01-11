from my_fun_new_colab import kernel


import cv2
import numpy as np
from scipy import ndimage


def refine(x, por):
    data = cv2.GaussianBlur(np.float32(x), (5, 5), 1)
    data = cv2.medianBlur(np.float32(data), 5)
    data = cv2.GaussianBlur(np.float32(data), (5, 5), 1)
    data = cv2.medianBlur(np.float32(data), 5)
    highpass_1 = ndimage.convolve(data, kernel)
    # зануление пикслей ниже заданного порога
    nmi = np.where(highpass_1 / 255 < por, np.zeros(x.shape), highpass_1)
    highpass_1 = 2 * nmi
    nma = np.where(highpass_1 / 255 > 1, 255 * np.ones(x.shape), highpass_1)
    highpass_1 = nma
    # if por==0:
    #     por_=0.2
    # else:
    #     por_=2*por
    # nma=np.where(highpass_1/255>por_)
    # highpass_1[nma]=255
    # highpass_1 = cv2.dilate(highpass_1.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)), iterations = 1)
    # highpass_1= highpass_1.astype('float32')
    # highpass_1[nma]=128
    return data, highpass_1


def pred_norm_img(img):
    img = img.astype("float32")
    hist = np.histogram(img, bins=255, density=True)
    a = hist[0]
    d = np.cumsum(a) / sum(a)
    s1 = np.argwhere(d >= 0.05)
    mi = s1[0]
    s2 = np.argwhere(d >= 0.999)
    ma = s2[0]

    # Нормировка  е единому диапазону
    img_ = (img - mi) / (ma - mi)
    a = np.where(img_ < 0)
    img_[a] = 0
    b = np.where(img_ > 1)
    img_[b] = 1
    img = (255 * img_).astype("uint8")
    hist = np.histogram(img, bins=255, density=True)
    a = hist[0]
    return img


def post_process(img_A, img_B, por_sum=5):
    contours_A, hierarchy = cv2.findContours(
        img_A, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # create an empty image for contours img_contours = np.zeros(img.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    img_contours_A = np.zeros(
        img_A.shape
    )  # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    cv2.drawContours(img_contours_A, contours_A, -1, (255, 0, 0), -1)
    cv2.imwrite("contours_A.png", img_contours_A)

    contours_B, hierarchy = cv2.findContours(
        img_B, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )  # create an empty image for contours img_contours = np.zeros(img.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    img_contours_B = np.zeros(
        img_B.shape
    )  # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    cv2.drawContours(img_contours_B, contours_B, -1, (255, 0, 0), -1)
    cv2.imwrite("contours_B.png", img_contours_B)
    res = np.zeros(len(contours_A))
    i = 0
    b = []
    while i < len(contours_A):
        img_contours_A_i = np.zeros(img_A.shape)
        # # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
        cv2.drawContours(img_contours_A_i, contours_A, i, (255, 0, 0), -1)
        res[i] = sum(sum(img_contours_A_i * img_contours_B)) / (255 * 255)
        if res[i] <= por_sum:
            del contours_A[i]
        else:
            i = i + 1

    img_contours_A_ = np.zeros(
        img_A.shape
    )  # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    cv2.drawContours(img_contours_A_, contours_A, -1, (255, 0, 0), -1)
    # cv2.imwrite('contours_A_.png',img_contours_A_)
    return img_contours_A_