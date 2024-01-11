from scipy import ndimage
from image_proccess import post_process, pred_norm_img, refine
from my_fun_new_colab import clahe, kernel


import cv2
import numpy as np


import os


def testModel_binary_AB(
    model, model_name, imgName, wsize, out_size, Ap_norm, Ap_clahe, hf, por
):  # непосредственно
    if os.path.exists("test_colab/" + model_name + "/") != True:
        #     os.remove('test_data/'+b)No documentation available
        os.makedirs("test_colab/" + model_name + "/")
    img_A = cv2.imread(imgName, 0)
    img_B = cv2.imread("train_data/94.png", 0)
    fix = model_name[0:8]  #
    # if Ap_clahe==True:
    #     img_A = clahe.apply(img_A)
    # if Ap_norm==True:
    #     img_A = pred_norm_img(img_A)
    # if Ap_norm==True and Ap_clahe==True:
    #     img_A = pred_norm_img(img_A)
    #     img_A = clahe.apply(img_A)
    if hf == "True":
        [img_A, img_B] = refine(img_A, por)

    elif hf == "Canny":
        img_B = cv2.Canny(img_A.astype("uint8"), 10, 100)

    width = img_A.shape[1]
    height = img_A.shape[0]
    blocksW = int(width / wsize)
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

    imgBin3_A = np.zeros((height, width, 1), np.uint8)
    imgBin3_B = np.zeros((height, width, 1), np.uint8)
    for x in range(blocksW):
        if x == 0:
            xmin = 0
            xmax = wsize
        elif x == 1:
            xmin = xmax - 2 * deltW
            xmax = xmin + wsize
        else:
            xmin = xmax - deltW
            xmax = xmin + wsize
        for y in range(blocksH):
            if y == 0:
                ymin = 0
                ymax = wsize
            elif y == 1:
                ymin = ymax - 2 * deltH
                ymax = ymin + wsize
            else:
                ymin = ymax - deltH
                ymax = ymin + wsize

            imgRoi_A = img_A[ymin:ymax, xmin:xmax]
            imgR_A = imgRoi_A.astype("float32") / 255.0
            imgRoi_B = img_B[ymin:ymax, xmin:xmax]
            imgR_B = imgRoi_B.astype("float32") / 255.0
            ptrnTestX_A = []
            ptrnTestX_A.append(imgR_A)
            ptrnTestX_A = np.array(ptrnTestX_A)
            ptrnTestX_A = ptrnTestX_A.reshape(ptrnTestX_A.shape[0], wsize, wsize, 1)
            ptrnTestX_B = []
            ptrnTestX_B.append(imgR_B)
            ptrnTestX_B = np.array(ptrnTestX_B)
            ptrnTestX_B = ptrnTestX_B.reshape(ptrnTestX_B.shape[0], wsize, wsize, 1)

            if fix == "Unet_V_0":
                prediction_A = model.predict(ptrnTestX_A)
                prediction_B = prediction_A
                prediction_A = (prediction_A * 255).astype("int8")
                prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                prediction_B = (prediction_B * 255).astype("int8")
                prediction_B = prediction_B.reshape(wsize, wsize, out_size)
            elif fix == "Unet_V_1" or fix == "Unet_V_4":
                prediction_A = model.predict([ptrnTestX_A, ptrnTestX_B])
                prediction_B = prediction_A
                prediction_A = (prediction_A * 255).astype("int8")
                prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                prediction_B = (prediction_B * 255).astype("int8")
                prediction_B = prediction_B.reshape(wsize, wsize, out_size)
            else:
                prediction_A, prediction_B = model.predict([ptrnTestX_A, ptrnTestX_B])
                prediction_A = (prediction_A * 255).astype("int8")
                prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                prediction_B = (prediction_B * 255).astype("int8")
                prediction_B = prediction_B.reshape(wsize, wsize, out_size)

            for col in range(wsize):
                for row in range(wsize):
                    if out_size == 1:
                        imgBin3_A[ymin + row, xmin + col, 0] = prediction_A[row, col, 0]
                        imgBin3_B[ymin + row, xmin + col, 0] = prediction_B[row, col, 0]
                    else:
                        if prediction_A[row, col, 1] < prediction_A[row, col, 0]:
                            imgBin3_A[ymin + row, xmin + col, 0] = 255
                        if prediction_B[row, col, 1] < prediction_B[row, col, 0]:
                            imgBin3_B[ymin + row, xmin + col, 0] = 255
    if fix == "Unet_V_0" or fix == "Unet_V_1" or fix == "Unet_V_4":
        if out_size == 1:
            tr, imgBin3_A = cv2.threshold(imgBin3_A, 127, 255, cv2.THRESH_BINARY)

    else:
        if out_size == 1:
            tr, imgBin3_A = cv2.threshold(imgBin3_A, 0.5, 1, cv2.THRESH_BINARY)
            tr, imgBin3_B = cv2.threshold(imgBin3_B, 127, 255, cv2.THRESH_BINARY)

        imgBin3_A = post_process(imgBin3_A, imgBin3_B, por_sum=10)

    print("img_A")
    a = imgName[11 : len(imgName) - 4]  # для переименования файла
    b = a + "_A.jpg"
    cv2.imwrite("test_colab/" + model_name + "/" + b, img_A)
    b = a + "_B.jpg"
    cv2.imwrite("test_colab/" + model_name + "/" + b, img_B)
    print("imgBin")
    b = a + "_3A.png"
    cv2.imwrite("test_colab/" + model_name + "/" + b, imgBin3_A)
    b = a + "_3B.png"
    cv2.imwrite("test_colab/" + model_name + "/" + b, imgBin3_B)
    return (img_A, img_B, imgBin3_A, imgBin3_B, prediction_A)


def testModel_binary_val_AB(
    model, model_name, basePath, wsize, out_size, Ap_norm, Ap_clahe, hf, por
):  # непосредственно
    """
    Создание выборки для валидации
    """
    fix = model_name[0:8]  #
    path_gray = basePath + "/gray 2014/1me"
    path_bin = basePath + "/binary 2014/1me"
    if os.path.exists("test_colab/" + model_name + "/") != True:
        #     os.remove('test_data/'+b)No documentation available
        os.makedirs("test_colab/" + model_name + "/")
    k = 0
    for fileName in os.listdir(path_gray):  # дает все что есть в каталоге
        k = k + 1
        # if (k-1) % 10 ==0: #остаток от деления
        if k % 5 != 0:
            # if (k-1) % 5 ==0: #остаток от деления
            img_A = cv2.imread(path_gray + "/" + fileName, 0)
            # Проверочный механизм
            img_B = cv2.imread(
                path_bin
                + "/"
                + fileName[0 : len(fileName) - 4]
                + " rbin [1;2;3;0;0;1;20;10;0;9;1;2].png",
                0,
            )
            if Ap_clahe == True:
                img_A = clahe.apply(img_A)
            if Ap_norm == True:
                img_A = pred_norm_img(img_A)
            if Ap_norm == True and Ap_clahe == True:
                img_A = pred_norm_img(img_A)
                img_A = clahe.apply(img_A)
            print("name ", fileName)

            if hf == "True":
                [img_A, img_B] = refine(img_A, por)
                # #Проверочный механизм
                # data= ndimage.convolve(img_B, kernel)
                # nmi=np.where(data<128)
                # data[nmi]=0
                # nma=np.where(data>=128)
                # data[nma]=255
                # img_B = cv2.dilate(data.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)), iterations = 1)
                # img_B= img_B.astype('float32')

            elif hf == "Canny":
                img_B = cv2.Canny(img_A.astype("uint8"), 10, 100)

            width = img_A.shape[1]
            height = img_A.shape[0]

            blocksW = int(width / wsize)
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

            imgBin3_A = np.zeros((height, width, 1), np.uint8)
            imgBin3_B = np.zeros((height, width, 1), np.uint8)
            for x in range(blocksW):
                if x == 0:
                    xmin = 0
                    xmax = wsize
                elif x == 1:
                    xmin = xmax - 2 * deltW
                    xmax = xmin + wsize
                else:
                    xmin = xmax - deltW
                    xmax = xmin + wsize
                for y in range(blocksH):
                    if y == 0:
                        ymin = 0
                        ymax = wsize
                    elif y == 1:
                        ymin = ymax - 2 * deltH
                        ymax = ymin + wsize
                    else:
                        ymin = ymax - deltH
                        ymax = ymin + wsize

                    imgRoi_A = img_A[ymin:ymax, xmin:xmax]
                    imgRoi_A = imgRoi_A.astype("float32") / 255.0
                    imgRoi_B = img_B[ymin:ymax, xmin:xmax]
                    imgRoi_B = imgRoi_B.astype("float32") / 255
                    ptrnTestX_A = []
                    ptrnTestX_A.append(imgRoi_A)
                    ptrnTestX_A = np.array(ptrnTestX_A)
                    ptrnTestX_A_ = ptrnTestX_A.reshape(
                        ptrnTestX_A.shape[0], wsize, wsize, 1
                    )
                    ptrnTestX_B = []
                    ptrnTestX_B.append(imgRoi_B)
                    ptrnTestX_B = np.array(ptrnTestX_B)
                    ptrnTestX_B_ = ptrnTestX_B.reshape(
                        ptrnTestX_B.shape[0], wsize, wsize, 1
                    )
                    if fix == "Unet_V_0":
                        prediction_A = model.predict(ptrnTestX_A)
                        prediction_B = prediction_A
                        prediction_A = (prediction_A * 255).astype("int8")
                        prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                        prediction_B = (prediction_B * 255).astype("int8")
                        prediction_B = prediction_B.reshape(wsize, wsize, out_size)
                    elif fix == "Unet_V_1" or fix == "Unet_V_4":
                        prediction_A = model.predict([ptrnTestX_A, ptrnTestX_B])
                        prediction_B = prediction_A
                        prediction_A = (prediction_A * 255).astype("int8")
                        prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                        prediction_B = (prediction_B * 255).astype("int8")
                        prediction_B = prediction_B.reshape(wsize, wsize, out_size)
                    else:
                        prediction_A, prediction_B = model.predict(
                            [ptrnTestX_A, ptrnTestX_B]
                        )
                        prediction_A = (prediction_A * 255).astype("int8")
                        prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                        prediction_B = (prediction_B * 255).astype("int8")
                        prediction_B = prediction_B.reshape(wsize, wsize, out_size)

                    for col in range(wsize):
                        for row in range(wsize):
                            if out_size == 1:
                                imgBin3_A[ymin + row, xmin + col, 0] = prediction_A[
                                    row, col, 0
                                ]
                                imgBin3_B[ymin + row, xmin + col, 0] = prediction_B[
                                    row, col, 0
                                ]
                            else:
                                if (
                                    prediction_A[row, col, 1]
                                    < prediction_A[row, col, 0]
                                ):
                                    imgBin3_A[ymin + row, xmin + col, 0] = 255
                                if (
                                    prediction_B[row, col, 1]
                                    < prediction_B[row, col, 0]
                                ):
                                    imgBin3_B[ymin + row, xmin + col, 0] = 255

            if fix == "Unet_V_0" or fix == "Unet_V_1" or fix == "Unet_V_4":
                if out_size == 1:
                    tr, imgBin3_A = cv2.threshold(
                        imgBin3_A, 127, 255, cv2.THRESH_BINARY
                    )
                    imgBin3_B = imgBin3_A
            else:
                if out_size == 1:
                    tr, imgBin3_A = cv2.threshold(imgBin3_A, 0.5, 1, cv2.THRESH_BINARY)
                    tr, imgBin3_B = cv2.threshold(
                        imgBin3_B, 127, 255, cv2.THRESH_BINARY
                    )

                imgBin3_A = post_process(imgBin3_A, imgBin3_B, por_sum=10)

            print("imgGray")
            a = fileName[0 : len(fileName) - 4]  # для переименования файла
            b = a + "_A.jpg"
            cv2.imwrite("test_colab/" + model_name + "/" + b, img_A)
            b = a + "_B.jpg"
            # cv2.imwrite('test_colab/'+model_name+'/'+b,img_B)
            print("imgBin")
            b = a + "_3A.png"
            cv2.imwrite("test_colab/" + model_name + "/" + b, imgBin3_A)
            b = a + "_3B.png"
            # cv2.imwrite('test_colab/'+model_name+'/'+b,imgBin3_B)
    return imgBin3_A, imgBin3_B


def testModel_binary(
    model, model_name, imgName, wsize, out_size, Ap_norm, Ap_clahe, hf
):  # непосредственно
    if os.path.exists("test_colab/" + model_name + "/") != True:
        #     os.remove('test_data/'+b)No documentation available
        os.makedirs("test_colab/" + model_name + "/")
    imgGray = cv2.imread(imgName, 0)
    data = np.array(imgGray, dtype="float32")
    data = cv2.GaussianBlur(data, (3, 3), 1)
    data = cv2.medianBlur(data, 5)
    if hf == "True":
        highpass_1 = ndimage.convolve(data, kernel)
        imgGray = data * 0.75 + highpass_1

    elif hf == "Canny":
        imgGray = cv2.Canny(data.astype("uint8"), 10, 100)

    width = imgGray.shape[1]
    height = imgGray.shape[0]

    # img_=(imgGray/255).astype('float32')
    # ss=0.1*np.random.random((height,width))
    # imgGray=((img_+ss)*255).astype('uint8')

    if Ap_clahe == True:
        imgGray = clahe.apply(imgGray)
    if Ap_norm == True:
        imgGray = pred_norm_img(imgGray)
    if Ap_norm == True and Ap_clahe == True:
        imgGray = pred_norm_img(imgGray)
        imgGray = clahe.apply(imgGray)

    blocksW = int(width / wsize)
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

    imgBin1 = np.zeros((height, width, 1), np.uint8)
    imgBin2 = np.zeros((height, width, 1), np.uint8)
    imgBin3 = np.zeros((height, width, 1), np.uint8)
    for x in range(blocksW):
        # x=1
        if x == 0:
            xmin = 0
            xmax = wsize
        elif x == 1:
            xmin = xmax - 2 * deltW
            xmax = xmin + wsize
        else:
            xmin = xmax - deltW
            xmax = xmin + wsize
        for y in range(blocksH):
            # y=1
            if y == 0:
                ymin = 0
                ymax = wsize
            elif y == 1:
                ymin = ymax - 2 * deltH
                ymax = ymin + wsize
            else:
                ymin = ymax - deltH
                ymax = ymin + wsize

            imgRoiGray = imgGray[ymin:ymax, xmin:xmax]
            imgRoiGray = imgRoiGray.astype("float32") / 255.0

            ptrnTestX = []
            ptrnTestX.append(imgRoiGray)
            ptrnTestX = np.array(ptrnTestX)
            ptrnTestX_ = ptrnTestX.reshape(ptrnTestX.shape[0], wsize, wsize, 1)

            prediction_ = model.predict(ptrnTestX_)
            prediction = (prediction_ * 255).astype("int8")
            prediction = prediction.reshape(wsize, wsize, out_size)

            if out_size == 1:
                # внимане второй индекс уменьшается на единицу
                imgBin1[ymin:ymax, xmin:xmax] = prediction
                imgBin2[ymin:ymax, xmin:xmax] = prediction
            elif out_size == 2:
                imgBin1[ymin:ymax, xmin:xmax, 0] = prediction[:, :, 0]
                imgBin2[ymin:ymax, xmin:xmax, 0] = prediction[:, :, 1]
                for col in range(wsize):
                    for row in range(wsize):
                        imgBin2[ymin + row, xmin + col, 0] = prediction[row, col, 1]
                        if prediction[row, col, 1] < prediction[row, col, 0]:
                            imgBin3[ymin + row, xmin + col, 0] = 255
    print("imgGray")
    a = imgName[11 : len(imgName) - 4]  # для переименования файла
    b = a + ".jpg"
    cv2.imwrite("test_colab/" + model_name + "/" + b, imgGray)
    print("imgBin")
    b = a + "_3.png"
    cv2.imwrite("test_colab/" + model_name + "/" + b, imgBin3)

    return (imgBin1, imgBin2, imgBin3, prediction)


def testModel_binary_val(
    model, model_name, basePath, wsize, out_size, Ap_norm, Ap_clahe, hf
):  # непосредственно
    """
    Создание выборки для валидации
    """
    path_gray = basePath + "/gray 2014/1me"
    path_BIN = basePath + "/gray 2014/train_colab_bin"
    if os.path.exists("test_colab/" + model_name + "/") != True:
        #     os.remove('test_data/'+b)No documentation available
        os.makedirs("test_colab/" + model_name + "/")
    k = 0
    for fileName in os.listdir(path_gray):  # дает все что есть в каталоге
        k = k + 1
        # if (k-1) % 10 ==0: #остаток от деления
        if (k - 1) % 5 == 0:  # остаток от деления
            imgGray = cv2.imread(path_gray + "/" + fileName, 0)
            fileName_ = fileName[0 : len(fileName) - 4] + "_3.jpg"
            imgBIN = cv2.imread(path_BIN + "/" + fileName_, 0)

            data = np.array(imgGray, dtype="float32")
            data = cv2.GaussianBlur(data, (3, 3), 1)
            data = cv2.medianBlur(data, 5)
            if hf == "True":
                highpass_1 = ndimage.convolve(data, kernel)
                imgGray = data * 0.75 + highpass_1

            elif hf == "Canny":
                imgGray = cv2.Canny(data.astype("uint8"), 10, 100)

            # data = np.array(imgGray, dtype='float32')
            # data = cv2.GaussianBlur(data,(5,5),cv2.BORDER_DEFAULT)
            # imgGray=data+imgBIN.astype('float32')

            width = imgGray.shape[1]
            height = imgGray.shape[0]

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

            blocksW = int(width / wsize)
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

            imgBin3 = np.zeros((height, width, 1), np.uint8)
            for x in range(blocksW):
                if x == 0:
                    xmin = 0
                    xmax = wsize
                elif x == 1:
                    xmin = xmax - 2 * deltW
                    xmax = xmin + wsize
                else:
                    xmin = xmax - deltW
                    xmax = xmin + wsize
                for y in range(blocksH):
                    # y=1
                    if y == 0:
                        ymin = 0
                        ymax = wsize
                    elif y == 1:
                        ymin = ymax - 2 * deltH
                        ymax = ymin + wsize
                    else:
                        ymin = ymax - deltH
                        ymax = ymin + wsize

                    imgRoiGray = imgGray[ymin:ymax, xmin:xmax]
                    imgRoiGray = imgRoiGray.astype("float32") / 255.0

                    ptrnTestX = []
                    ptrnTestX.append(imgRoiGray)
                    ptrnTestX = np.array(ptrnTestX)
                    ptrnTestX_ = ptrnTestX.reshape(ptrnTestX.shape[0], wsize, wsize, 1)

                    prediction_ = model.predict(ptrnTestX_)
                    prediction = (prediction_ * 255).astype("int8")
                    prediction = prediction.reshape(wsize, wsize, out_size)

                    if out_size == 1:
                        # внимане второй индекс уменьшается на единицу
                        imgBin3[ymin:ymax, xmin:xmax, 0] = prediction
                    elif out_size == 2:
                        for col in range(wsize):
                            for row in range(wsize):
                                if prediction[row, col, 1] < prediction[row, col, 0]:
                                    imgBin3[ymin + row, xmin + col, 0] = 255
            print("imgGray")
            a = fileName[0 : len(fileName) - 4]  # для переименования файла
            b = a + ".jpg"
            cv2.imwrite("test_colab/" + model_name + "/" + b, imgGray)
            print("imgBin")
            b = a + "_3.png"
            cv2.imwrite("test_colab/" + model_name + "/" + b, imgBin3)
    return imgBin3
