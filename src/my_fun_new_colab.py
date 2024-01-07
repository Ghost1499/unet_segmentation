#Импорт всех необходимых компонентов и утилит
#Установка библиотеки TensorFlow-Advanced-Segmentation-Models с разными функциями ошибок и сегментационными моделями

import cv2 #это opencv
import os
import numpy as np
import keras
from keras import backend as K #Для программирования собственных метрик
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from keras.layers import Layer,Add,Conv2D, Activation, Dropout,Dense
from keras.layers import Input,MaxPooling2D,UpSampling2D,concatenate
from keras.models import Sequential, load_model, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D
from tensorflow.python.keras import layers
from keras import optimizers
from keras import regularizers
from keras.regularizers import l2
from keras.metrics import Precision,Recall
from scipy import ndimage
def shuffle_unson_write_AB(basePath,end_path,ptrnsTrainX_A,ptrnsTrainX_B,ptrnsTrainY_A,ptrnsTrainY_B):
    #Перемешивание данных для последдующей записи
    a=ptrnsTrainX_A.shape
    b=ptrnsTrainY_A.shape
    assert a[0] == b[0]
    np.random.seed(10)
    p = np.random.permutation(a[0])
    
    ptrnsTrainX_A_=ptrnsTrainX_A[p,:,:,:]
    ptrnsTrainX_B_=ptrnsTrainX_B[p,:,:,:]
    ptrnsTrainY_A_=ptrnsTrainY_A[p,:,:,:]
    ptrnsTrainY_B_=ptrnsTrainY_B[p,:,:,:]
    #end_path='_single_'+str(wsize)+'_'+str(out_size)+'_'+str(Ap_norm)+'_'+str(Ap_clahe)+'_'+str(por)
    if basePath=='train_colab':
        np.save(basePath+'/gray/ptrnsTrainX_AB'+end_path+'.npy',(ptrnsTrainX_A_,ptrnsTrainX_B_))
        np.save(basePath+'/binary/ptrnsTrainY_AB'+end_path+'.npy',(ptrnsTrainY_A_,ptrnsTrainY_B_))
    elif basePath=='val_colab':
        np.save(basePath+'/gray/ptrnsValX_AB'+end_path+'.npy',(ptrnsTrainX_A_,ptrnsTrainX_B_))
        #ptrnsTrainX=np.load(basePath+'/gray/ptrnsTrainX.npy');#проверка
        np.save(basePath+'/binary/ptrnsValY_AB'+end_path+'.npy',(ptrnsTrainY_A_,ptrnsTrainY_B_))
   
def F1(y_true, y_pred):
    pr = tf.keras.metrics.Precision()
    pr.update_state(y_true, y_pred)
    re = tf.keras.metrics.Recall()
    re.update_state(y_true, y_pred)
    pr_=pr.result().numpy()
    re_=re.result().numpy()
    f1_score = 2 * (pr_ * re_) / (pr_ + re_)
    return f1_score
y_true, y_pred=[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0] 
s=F1(y_true, y_pred)
def f1_metric(y_true_, y_pred_):
    y_true = np.array(y_true_)
    y_pred = np.array(y_pred_)
    true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val 
# y_true=np.array([0, 0, 1, 1]);y_pred=np.array([1, 1, 1, 1])
s=f1_metric(y_true, y_pred)


clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(32,32))#адаптивная эквализация
#tf.config.run_functions_eagerly(True)
kernel = np.array([[-1, -1, -1, -1, -1],
                   [-1,  1,  2,  1, -1],
                   [-1,  2,  4,  2, -1],
                   [-1,  1,  2,  1, -1],
                   [-1, -1, -1, -1, -1]])
def refine(x,por):
    data = cv2.GaussianBlur(np.float32(x),(5,5),1)
    data = cv2.medianBlur(np.float32(data),5)
    data = cv2.GaussianBlur(np.float32(data),(5,5),1)
    data = cv2.medianBlur(np.float32(data),5)
    highpass_1 = ndimage.convolve(data, kernel)
    #зануление пикслей ниже заданного порога
    nmi=np.where(highpass_1/255<por,np.zeros(x.shape),highpass_1)
    highpass_1=2*nmi
    nma=np.where(highpass_1/255>1,255*np.ones(x.shape),highpass_1)
    highpass_1=nma
    # if por==0:
    #     por_=0.2
    # else:
    #     por_=2*por
    # nma=np.where(highpass_1/255>por_)
    # highpass_1[nma]=255
    # highpass_1 = cv2.dilate(highpass_1.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)), iterations = 1)
    # highpass_1= highpass_1.astype('float32')
    #highpass_1[nma]=128
    return data,highpass_1

def pred_norm_img(img):
    img=img.astype('float32')
    hist=np.histogram(img,bins=255,density=True)
    a=hist[0]
    d=np.cumsum(a)/sum(a) 
    s1=np.argwhere(d >= 0.05)
    mi=s1[0]
    s2=np.argwhere(d >= 0.999)
    ma=s2[0]
       
    #Нормировка  е единому диапазону
    img_=(img-mi)/(ma-mi)
    a=np.where(img_<0); img_[a]=0
    b=np.where(img_>1); img_[b]=1
    img=(255*img_).astype('uint8')
    hist=np.histogram(img,bins=255,density=True)
    a=hist[0]
    return img
def post_process(img_A,img_B,por_sum=5):
    contours_A, hierarchy = cv2.findContours(img_A, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #create an empty image for contours img_contours = np.zeros(img.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    img_contours_A = np.zeros(img_A.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    cv2.drawContours(img_contours_A,contours_A,-1,(255,0,0),-1)
    cv2.imwrite('contours_A.png',img_contours_A)
        
    contours_B, hierarchy = cv2.findContours(img_B,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #create an empty image for contours img_contours = np.zeros(img.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    img_contours_B = np.zeros(img_B.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    cv2.drawContours(img_contours_B,contours_B,-1,(255,0,0),-1)
    cv2.imwrite('contours_B.png',img_contours_B)
    res=np.zeros(len(contours_A))
    i=0
    b = []
    while i<len(contours_A):
        img_contours_A_i = np.zeros(img_A.shape)
        # # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)
        cv2.drawContours(img_contours_A_i,contours_A,i,(255,0,0),-1)
        res[i]=sum(sum(img_contours_A_i*img_contours_B))/(255*255)
        if res[i]<=por_sum:
            del contours_A[i]
        else:     
            i=i+1
             
    img_contours_A_ = np.zeros(img_A.shape) # draw the contours on the empty image cv2.drawContours(img_contours, contours, -1, (0,255,0), 3) #save image cv2.imwrite('D:/contours.png',img_contours)
    cv2.drawContours(img_contours_A_,contours_A,-1,(255,0,0),-1)
    #cv2.imwrite('contours_A_.png',img_contours_A_)
    return img_contours_A_

def createPtrns_Gray_AB(img_A,wsize,ptrnTrain_A,ptrnTrain_B,hf,por):
    if hf=='True':
        [img_A,img_B]=refine(img_A,por)
                
    elif hf=='Canny':
        img_B = cv2.Canny(img_A.astype('uint8'),10,100)
    width = img_A.shape[1] #размеры входного изображения (по X)
    height = img_A.shape[0] #размеры входного изображения (по Y)
    imgRoi_A=np.zeros((wsize,wsize),np.float32)
    imgRoi_B=np.zeros((wsize,wsize),np.float32)
    blocksW = int(width/wsize) #число блоков
    blocksH = int(height/wsize)
    deltW=0
    deltH=0
    if width % wsize!=0:
        blocksW =blocksW +1
        width_=blocksW*wsize
        deltW=int((width_-width)/blocksW)
        
    if height % wsize!=0:
        blocksH = blocksH+1 #
        height_=blocksH*wsize
        deltH=int((height_- height)/blocksH)
    
    for x in range(blocksW): #*2):
        if x==0:
            xmin=0; xmax=wsize
        elif x==1:
            xmin=xmax-2*deltW; xmax=xmin+wsize
        else: 
            xmin=xmax-deltW; xmax=xmin+wsize
        # print(xmin,xmax)
        for y in range(blocksH): #*2):
            if y==0:
                ymin=0; ymax=wsize
            elif y==1:
                ymin=ymax-2*deltH; ymax=ymin+wsize
            else:
                ymin=ymax-deltH; ymax=ymin+wsize
            # print(ymin,ymax)
                
            imgRoi_A = img_A[ymin : ymax, xmin : xmax]
            imgRoi_A = imgRoi_A.astype('float32')/255.0
            imgRoi_B = img_B[ymin : ymax, xmin : xmax]
            imgRoi_B = imgRoi_B.astype('float32')/255.0
            ptrnTrain_A.append(imgRoi_A) #добавляет объект в конце списка
            ptrnTrain_B.append(imgRoi_B) 
def createPtrns_Bin_AB(img_A, wsize,ptrnTrain_A,ptrnTrain_B):
    data = np.array(img_A, dtype='float32')
    # data = cv2.GaussianBlur(data,(3,3),1)
    # data = cv2.medianBlur(data, 5)
    data= ndimage.convolve(data, kernel)
    nmi=np.where(data<128)
    data[nmi]=0
    nma=np.where(data>=128)
    data[nma]=255
    img_B = cv2.dilate(data.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)), iterations = 1)
    img_B= img_B.astype('float32')
    width = img_A.shape[1] #размеры входного изображения (по X)
    height = img_A.shape[0] #размеры входного изображения (по Y)
    imgRoi_A=np.zeros((wsize,wsize),np.float32)
    imgRoi_B=np.zeros((wsize,wsize),np.float32)
    blocksW = int(width/wsize) #число блоков
    blocksH = int(height/wsize)
    deltW=0
    deltH=0
    if width % wsize!=0:
        blocksW =blocksW +1
        width_=blocksW*wsize
        deltW=int((width_-width)/blocksW)
        
    if height % wsize!=0:
        blocksH = blocksH+1 #
        height_=blocksH*wsize
        deltH=int((height_- height)/blocksH)
    #step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
    for x in range(blocksW): #*2):
        if x==0:
            xmin=0; xmax=wsize
        elif x==1:
            xmin=xmax-2*deltW; xmax=xmin+wsize
        else: 
            xmin=xmax-deltW; xmax=xmin+wsize
        # print(xmin,xmax)
        for y in range(blocksH): #*2):
            if y==0:
                ymin=0; ymax=wsize
            elif y==1:
                ymin=ymax-2*deltH; ymax=ymin+wsize
            else:
                ymin=ymax-deltH; ymax=ymin+wsize
            # print(ymin,ymax)
                
            imgRoi_A = img_A[ymin : ymax, xmin : xmax]
            imgRoi_A = imgRoi_A.astype('float32') / 255.0
            ptrnTrain_A.append(imgRoi_A) #добавляет объект в
            imgRoi_B = img_B[ymin : ymax, xmin : xmax]
            imgRoi_B = imgRoi_B.astype('float32') / 255.0
            ptrnTrain_B.append(imgRoi_B)
#Опредление функции для загрузки данных из папки с заданным путем basePath
#и формирования обучающей выборки с нарезкой
def loadTrainData_binary_colab_AB(basePath,wsize,Ap_norm,Ap_clahe,hf,por):
    #out_size==2
    ptrnsTrainX_A = []
    ptrnsTrainX_B = []
    ptrnsTrainY_A = []
    ptrnsTrainY_B = []
    #смешанные данные
    # path_gray = basePath + "/gray_mixt"
    # path_binary = basePath + "/binary_mixt"
    #данные из одного слоя
    # path_gray = basePath +'/gray 2014/1me'
    path_gray = basePath +'/gray 2014/1me'
    path_binary = basePath + '/binary 2014/1me'
    k=0
    l=0
    #os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри 
    for fileName in os.listdir(path_gray):#дает все что есть в каталоге
        k=k+1;
        if k % 5==0: #остаток от деления
            l=l+1
            imgGray = cv2.imread(path_gray + '/' + fileName, 0)
            width = imgGray.shape[1] #размеры входного изображения (по X)
            height = imgGray.shape[0] 
            
            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((height,width))
            # imgGray=((img_+ss)*255).astype('uint8')
            
            if Ap_clahe==True and Ap_norm==False:
                imgGray = clahe.apply(imgGray)
            if Ap_clahe==False and Ap_norm==True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm==True and Ap_clahe==True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print('name ', fileName)
                        
            createPtrns_Gray_AB(imgGray, wsize, ptrnsTrainX_A,ptrnsTrainX_B,hf,por)
            imgGray_=imgGray.T
            createPtrns_Gray_AB(imgGray_, wsize,ptrnsTrainX_A,ptrnsTrainX_B,hf,por)
    print(l)
    
    k=0
    l=0
    for fileName in os.listdir(path_binary):
        k=k+1
        if k % 5==0:
            l=l+1
            imgBin = cv2.imread(path_binary + '/' + fileName, 0)
            print('name ', fileName)
            createPtrns_Bin_AB(imgBin,wsize,ptrnsTrainY_A,ptrnsTrainY_B)
            imgBin_=imgBin.T
            createPtrns_Bin_AB(imgBin_,wsize,ptrnsTrainY_A,ptrnsTrainY_B)
    print(l)

    ptrnsTrainX_A = np.array(ptrnsTrainX_A)
    ptrnsTrainX_A = ptrnsTrainX_A.reshape(ptrnsTrainX_A.shape[0], wsize, wsize, 1) #3
    ptrnsTrainX_B = np.array(ptrnsTrainX_B)
    ptrnsTrainX_B = ptrnsTrainX_B.reshape(ptrnsTrainX_B.shape[0], wsize, wsize, 1) #3
    
        
    #развертка в матрицу изображения
    ptrnsTrainY_A = np.array(ptrnsTrainY_A)
    ptrnsTrainY_A = ptrnsTrainY_A.reshape(ptrnsTrainY_A.shape[0], wsize, wsize,1) #3
    ptrnsTrainY_B = np.array(ptrnsTrainY_B)
    ptrnsTrainY_B = ptrnsTrainY_B.reshape(ptrnsTrainY_B.shape[0], wsize, wsize,1) #3
    return (ptrnsTrainX_A,ptrnsTrainX_B,ptrnsTrainY_A,ptrnsTrainY_B)
#Создание выборки для валидации
def loadValData_binary_colab_AB(basePath,wsize,Ap_norm,Ap_clahe,hf,por):
    ptrnsValX_A = []
    ptrnsValX_B = []
    ptrnsValY_A = []
    ptrnsValY_B = []
    #смешанные данные
    # path_gray = basePath + "/gray_mixt"
    # path_binary = basePath + "/binary_mixt"
    #данные из одного слоя
    # path_gray = basePath +'/gray 2014/1me'
    path_gray = basePath +'/gray 2014/1me'
    path_binary = basePath + '/binary 2014/1me'
    k=0
    l=0
    #os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри 
    for fileName in os.listdir(path_gray):#дает все что есть в каталоге
        k=k+1
        if (k-1) % 10 ==0:
            l=l+1
            imgGray = cv2.imread(path_gray + '/' + fileName, 0)
            width = imgGray.shape[1] #размеры входного изображения (по X)
            height = imgGray.shape[0] 
                    
            if Ap_clahe==True and Ap_norm==False:
                imgGray = clahe.apply(imgGray)
            if Ap_clahe==False and Ap_norm==True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm==True and Ap_clahe==True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print('name ', fileName)
            
            createPtrns_Gray_AB(imgGray, wsize, ptrnsValX_A,ptrnsValX_B,hf,por)
            imgGray_=imgGray.T
            createPtrns_Gray_AB(imgGray_, wsize, ptrnsValX_A,ptrnsValX_B,hf,por)
    print(l)
    
    k=0
    l=0
    for fileName in os.listdir(path_binary):
        k=k+1
        if (k-1) % 10==0:
            l=l+1
            imgBin = cv2.imread(path_binary + '/' + fileName, 0)
            print('name ', fileName)
            createPtrns_Bin_AB(imgBin,wsize,ptrnsValY_A,ptrnsValY_B)
            imgBin_=imgBin.T
            createPtrns_Bin_AB(imgBin_, wsize, ptrnsValY_A,ptrnsValY_B)
    print(l)    
    ptrnsValX_A = np.array(ptrnsValX_A)
    ptrnsValX_A = ptrnsValX_A.reshape(ptrnsValX_A.shape[0], wsize, wsize, 1) #3
    ptrnsValX_B = np.array(ptrnsValX_B)
    ptrnsValX_B = ptrnsValX_B.reshape(ptrnsValX_B.shape[0], wsize, wsize, 1) #3
    ptrnsValY_A = np.array(ptrnsValY_A)
    ptrnsValY_A = ptrnsValY_A.reshape(ptrnsValX_A.shape[0], wsize, wsize, 1) #3
    ptrnsValY_B = np.array(ptrnsValY_B)
    ptrnsValY_B = ptrnsValY_B.reshape(ptrnsValX_B.shape[0], wsize, wsize, 1) #3   
    return (ptrnsValX_A,ptrnsValX_B,ptrnsValY_A,ptrnsValY_B)

def testModel_binary_AB(model,model_name,imgName,wsize,out_size,Ap_norm,Ap_clahe,hf,por): #непосредственно
    if os.path.exists('test_colab/'+model_name+'/')!=True:
        #     os.remove('test_data/'+b)No documentation available 
        os.makedirs('test_colab/'+model_name+'/')
    img_A = cv2.imread(imgName,0)
    img_B=cv2.imread('train_data/94.png', 0)
    fix=model_name[0:8] #
    # if Ap_clahe==True:
    #     img_A = clahe.apply(img_A)
    # if Ap_norm==True:
    #     img_A = pred_norm_img(img_A)
    # if Ap_norm==True and Ap_clahe==True:
    #     img_A = pred_norm_img(img_A)
    #     img_A = clahe.apply(img_A)
    if hf=='True':
        [img_A,img_B]=refine(img_A,por)
        
        
               
    elif hf=='Canny':
        img_B = cv2.Canny(img_A.astype('uint8'),10,100)
        
    width = img_A.shape[1]
    height = img_A.shape[0]
    blocksW = int(width/wsize)
    blocksH = int(height/wsize)
    deltW=0
    deltH=0
    if width % wsize!=0:
        blocksW =blocksW +1
        width_=blocksW*wsize
        deltW=int((width_-width)/blocksW)
        
    if height % wsize!=0:
        blocksH = blocksH+1 #
        height_=blocksH*wsize
        deltH=int((height_- height)/blocksH)
    #step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
         
    
    imgBin3_A = np.zeros((height,width,1), np.uint8)
    imgBin3_B = np.zeros((height,width,1), np.uint8)
    for x in range(blocksW):
        if x==0:
            xmin=0; xmax=wsize
        elif x==1:
            xmin=xmax-2*deltW; xmax=xmin+wsize
        else: 
            xmin=xmax-deltW; xmax=xmin+wsize
        for y in range(blocksH):
            if y==0:
                ymin=0; ymax=wsize
            elif y==1:
                ymin=ymax-2*deltH; ymax=ymin+wsize
            else:
                ymin=ymax-deltH; ymax=ymin+wsize
            
            imgRoi_A = img_A[ymin : ymax, xmin : xmax]
            imgR_A = imgRoi_A.astype('float32') / 255.0
            imgRoi_B = img_B[ymin : ymax, xmin : xmax]
            imgR_B = imgRoi_B.astype('float32') / 255.0
            ptrnTestX_A = []
            ptrnTestX_A.append(imgR_A)
            ptrnTestX_A = np.array(ptrnTestX_A)
            ptrnTestX_A = ptrnTestX_A.reshape(ptrnTestX_A.shape[0], wsize, wsize, 1)
            ptrnTestX_B = []
            ptrnTestX_B.append(imgR_B)
            ptrnTestX_B = np.array(ptrnTestX_B)
            ptrnTestX_B = ptrnTestX_B.reshape(ptrnTestX_B.shape[0], wsize, wsize, 1)
            
            if fix== 'Unet_V_0':
                prediction_A = model.predict(ptrnTestX_A)
                prediction_B =prediction_A
                prediction_A = (prediction_A*255).astype('int8')
                prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                prediction_B = (prediction_B*255).astype('int8')
                prediction_B = prediction_B.reshape(wsize, wsize, out_size)
            elif fix== 'Unet_V_1'or fix=='Unet_V_4':     
                prediction_A = model.predict([ptrnTestX_A,ptrnTestX_B])
                prediction_B =prediction_A
                prediction_A = (prediction_A*255).astype('int8')
                prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                prediction_B = (prediction_B*255).astype('int8')
                prediction_B = prediction_B.reshape(wsize, wsize, out_size)            
            else:
                prediction_A,prediction_B = model.predict([ptrnTestX_A,ptrnTestX_B])
                prediction_A = (prediction_A*255).astype('int8')
                prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                prediction_B = (prediction_B*255).astype('int8')
                prediction_B = prediction_B.reshape(wsize, wsize, out_size)                           
           
            for col in range(wsize):
               for row in range(wsize):
                   if out_size==1:
                       imgBin3_A[ymin + row, xmin + col,0]=prediction_A[row, col, 0]
                       imgBin3_B[ymin + row,xmin + col,0]= prediction_B[row, col, 0]
                   else:
                        if prediction_A[row, col,1]<prediction_A[row, col, 0]:
                            imgBin3_A[ymin + row, xmin + col,0]=255
                        if prediction_B[row, col,1]<prediction_B[row, col, 0]:
                            imgBin3_B[ymin + row, xmin + col,0]=255
    if fix=='Unet_V_0' or fix=='Unet_V_1'or fix=='Unet_V_4':
         if out_size==1:
            tr,imgBin3_A = cv2.threshold(imgBin3_A,127,255,cv2.THRESH_BINARY)
    
    else:
        if out_size==1:
            tr,imgBin3_A = cv2.threshold(imgBin3_A,0.5,1,cv2.THRESH_BINARY)
            tr,imgBin3_B = cv2.threshold(imgBin3_B,127,255,cv2.THRESH_BINARY)
            
        imgBin3_A=post_process(imgBin3_A,imgBin3_B,por_sum=10)
    
    
    print('img_A')
    a=imgName[11:len(imgName)-4] #для переименования файла
    b=a+'_A.jpg'
    cv2.imwrite('test_colab/'+model_name+'/'+b,img_A)
    b=a+'_B.jpg'
    cv2.imwrite('test_colab/'+model_name+'/'+b,img_B)
    print('imgBin')
    b=a+'_3A.png'
    cv2.imwrite('test_colab/'+model_name+'/'+b,imgBin3_A)
    b=a+'_3B.png'
    cv2.imwrite('test_colab/'+model_name+'/'+b,imgBin3_B)
    return (img_A,img_B,imgBin3_A,imgBin3_B,prediction_A)            
#Создание выборки для валидации
def testModel_binary_val_AB(model,model_name,basePath,wsize,out_size,Ap_norm,Ap_clahe,hf,por): #непосредственно
    fix=model_name[0:8] #
    path_gray = basePath +'/gray 2014/1me'
    path_bin=basePath +'/binary 2014/1me'
    if os.path.exists('test_colab/'+model_name+'/')!=True:
        #     os.remove('test_data/'+b)No documentation available 
        os.makedirs('test_colab/'+model_name+'/')
    k=0     
    for fileName in os.listdir(path_gray):#дает все что есть в каталоге
        k=k+1;
        # if (k-1) % 10 ==0: #остаток от деления
        if k % 5 !=0:
        #if (k-1) % 5 ==0: #остаток от деления
            img_A = cv2.imread(path_gray + '/' + fileName, 0)
             #Проверочный механизм
            img_B=cv2.imread(path_bin + '/' + fileName[0:len(fileName)-4]+' rbin [1;2;3;0;0;1;20;10;0;9;1;2].png', 0)
            if Ap_clahe==True:
                img_A = clahe.apply(img_A)
            if Ap_norm==True:
                img_A = pred_norm_img(img_A)
            if Ap_norm==True and Ap_clahe==True:
                img_A = pred_norm_img(img_A)
                img_A = clahe.apply(img_A)
            print('name ',fileName)
            
            
            if hf=='True':
                [img_A,img_B]=refine(img_A,por)
                # #Проверочный механизм
                # data= ndimage.convolve(img_B, kernel)
                # nmi=np.where(data<128)
                # data[nmi]=0
                # nma=np.where(data>=128)
                # data[nma]=255
                # img_B = cv2.dilate(data.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2)), iterations = 1)
                # img_B= img_B.astype('float32')
                
               
            elif hf=='Canny':
                img_B = cv2.Canny(img_A.astype('uint8'),10,100)
            
            width = img_A.shape[1]
            height = img_A.shape[0]

            blocksW = int(width/wsize)
            blocksH = int(height/wsize)
            deltW=0
            deltH=0
            if width % wsize!=0:
                blocksW =blocksW +1
                width_=blocksW*wsize
                deltW=int((width_-width)/blocksW)
            if height % wsize!=0:
                blocksH = blocksH+1 #
                height_=blocksH*wsize
                deltH=int((height_- height)/blocksH)
                                    
            imgBin3_A = np.zeros((height,width,1), np.uint8)
            imgBin3_B = np.zeros((height,width,1), np.uint8)
            for x in range(blocksW):
                if x==0:
                    xmin=0; xmax=wsize
                elif x==1:
                    xmin=xmax-2*deltW; xmax=xmin+wsize
                else: 
                    xmin=xmax-deltW; xmax=xmin+wsize
                for y in range(blocksH):
                    if y==0:
                        ymin=0; ymax=wsize
                    elif y==1:
                        ymin=ymax-2*deltH; ymax=ymin+wsize
                    else:
                        ymin=ymax-deltH; ymax=ymin+wsize
                    
                    imgRoi_A = img_A[ymin : ymax, xmin : xmax]
                    imgRoi_A = imgRoi_A.astype('float32') / 255.0
                    imgRoi_B = img_B[ymin : ymax, xmin : xmax]
                    imgRoi_B = imgRoi_B.astype('float32') / 255
                    ptrnTestX_A = []
                    ptrnTestX_A.append(imgRoi_A)
                    ptrnTestX_A = np.array(ptrnTestX_A)
                    ptrnTestX_A_ = ptrnTestX_A.reshape(ptrnTestX_A.shape[0], wsize, wsize, 1)
                    ptrnTestX_B = []
                    ptrnTestX_B.append(imgRoi_B)
                    ptrnTestX_B = np.array(ptrnTestX_B)
                    ptrnTestX_B_ = ptrnTestX_B.reshape(ptrnTestX_B.shape[0], wsize, wsize, 1)                
                    if fix== 'Unet_V_0':
                        prediction_A = model.predict(ptrnTestX_A)
                        prediction_B =prediction_A
                        prediction_A = (prediction_A*255).astype('int8')
                        prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                        prediction_B = (prediction_B*255).astype('int8')
                        prediction_B = prediction_B.reshape(wsize, wsize, out_size)
                    elif fix== 'Unet_V_1'or fix=='Unet_V_4':     
                        prediction_A = model.predict([ptrnTestX_A,ptrnTestX_B])
                        prediction_B =prediction_A
                        prediction_A = (prediction_A*255).astype('int8')
                        prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                        prediction_B = (prediction_B*255).astype('int8')
                        prediction_B = prediction_B.reshape(wsize, wsize, out_size)            
                    else:
                        prediction_A,prediction_B = model.predict([ptrnTestX_A,ptrnTestX_B])
                        prediction_A = (prediction_A*255).astype('int8')
                        prediction_A = prediction_A.reshape(wsize, wsize, out_size)
                        prediction_B = (prediction_B*255).astype('int8')
                        prediction_B = prediction_B.reshape(wsize, wsize, out_size)                           
                   
                    
                    for col in range(wsize):
                       for row in range(wsize):
                           if out_size==1:
                               imgBin3_A[ymin + row, xmin + col,0]= prediction_A[row, col, 0]
                               imgBin3_B[ymin + row, xmin + col,0]= prediction_B[row, col, 0]
                           else:
                                if prediction_A[row, col,1]<prediction_A[row, col, 0]:
                                    imgBin3_A[ymin + row, xmin + col,0]=255
                                if prediction_B[row, col,1]<prediction_B[row, col, 0]:
                                    imgBin3_B[ymin + row, xmin + col,0]=255
            
            if fix=='Unet_V_0' or fix=='Unet_V_1' or fix=='Unet_V_4':
                 if out_size==1:
                    tr,imgBin3_A = cv2.threshold(imgBin3_A,127,255,cv2.THRESH_BINARY)
                    imgBin3_B=imgBin3_A 
            else:
                if out_size==1:
                    tr,imgBin3_A = cv2.threshold(imgBin3_A,0.5,1,cv2.THRESH_BINARY)
                    tr,imgBin3_B = cv2.threshold(imgBin3_B,127,255,cv2.THRESH_BINARY)
                    
                imgBin3_A=post_process(imgBin3_A,imgBin3_B,por_sum=10)
            
           
                
            print('imgGray')
            a=fileName[0:len(fileName)-4] #для переименования файла
            b=a+'_A.jpg'
            cv2.imwrite('test_colab/'+model_name+'/'+b,img_A)
            b=a+'_B.jpg'
            #cv2.imwrite('test_colab/'+model_name+'/'+b,img_B)
            print('imgBin')
            b=a+'_3A.png'
            cv2.imwrite('test_colab/'+model_name+'/'+b,imgBin3_A)
            b=a+'_3B.png'
            #cv2.imwrite('test_colab/'+model_name+'/'+b,imgBin3_B)
    return imgBin3_A,imgBin3_B
#и формирования тестирующей выборки с нарезкой
def loadTestData_binary_colab_AB(basePath,wsize,Ap_norm,Ap_clahe,hf,por):
    #out_size==2
    ptrnsTestX_A = []
    ptrnsTestX_B = []
    ptrnsTestY_A = []
    ptrnsTestY_B = []
    path_gray = basePath +'/gray 2014/1me'
        
    k=0;
    #os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри 
    for fileName in os.listdir(path_gray):#дает все что есть в каталоге
        k=k+1;
        if k % 5 !=0: #остаток от деления
            imgGray = cv2.imread(path_gray + '/' + fileName, 0)
                                 
            if Ap_clahe==True:
                imgGray = clahe.apply(imgGray)
            if Ap_norm==True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm==True and Ap_clahe==True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print('name ', fileName)
            createPtrns_Gray_AB(imgGray, wsize, ptrnsTestX_A,ptrnsTestX_B,hf,por)
            imgGray_=imgGray.T
            createPtrns_Gray_AB(imgGray_, wsize, ptrnsTestX_A,ptrnsTestX_B,hf,por)
    path = basePath + "/binary 2014/1me"
    k=0;
    for fileName in os.listdir(path):
        k=k+1
        if k % 5 !=0:
            imgBin = cv2.imread(path + '/' + fileName, 0)
            print('name ', fileName)
            createPtrns_Bin_AB(imgBin, wsize, ptrnsTestY_A,ptrnsTestY_B)
            imgBin_=imgBin.T
            createPtrns_Bin_AB(imgBin_, wsize, ptrnsTestY_A,ptrnsTestY_B)

    ptrnsTestX_A = np.array(ptrnsTestX_A)
    ptrnsTestX_A = ptrnsTestX_A.reshape(ptrnsTestX_A.shape[0], wsize, wsize, 1) #3
    ptrnsTestX_B = np.array(ptrnsTestX_B)
    ptrnsTestX_B = ptrnsTestX_B.reshape(ptrnsTestX_B.shape[0], wsize, wsize, 1) #3
        
    #развертка в матрицу изображения
    ptrnsTestY_A = np.array(ptrnsTestY_A)
    ptrnsTestY_A = ptrnsTestY_A.reshape(ptrnsTestY_A.shape[0], wsize, wsize,1) #3
    ptrnsTestY_B = np.array(ptrnsTestY_B)
    ptrnsTestY_B = ptrnsTestY_B.reshape(ptrnsTestY_B.shape[0], wsize, wsize,1) #3
    return (ptrnsTestX_A,ptrnsTestX_B,ptrnsTestY_A,ptrnsTestY_B)    

def shuffle_unson_write_AB(basePath,end_path,ptrnsTrainX_A,ptrnsTrainX_B,ptrnsTrainY_A,ptrnsTrainY_B):
    #Перемешивание данных для последдующей записи
    a=ptrnsTrainX_A.shape
    b=ptrnsTrainY_A.shape
    assert a[0] == b[0]
    np.random.seed(10)
    p = np.random.permutation(a[0])
    
    ptrnsTrainX_A_=ptrnsTrainX_A[p,:,:,:]
    ptrnsTrainX_B_=ptrnsTrainX_B[p,:,:,:]
    ptrnsTrainY_A_=ptrnsTrainY_A[p,:,:,:]
    ptrnsTrainY_B_=ptrnsTrainY_B[p,:,:,:]
    #end_path='_single_'+str(wsize)+'_'+str(out_size)+'_'+str(Ap_norm)+'_'+str(Ap_clahe)+'_'+str(por)
    if basePath=='train_colab':
        np.save(basePath+'/gray/ptrnsTrainX_AB'+end_path+'.npy',(ptrnsTrainX_A_,ptrnsTrainX_B_))
        np.save(basePath+'/binary/ptrnsTrainY_AB'+end_path+'.npy',(ptrnsTrainY_A_,ptrnsTrainY_B_))
    elif basePath=='val_colab':
        np.save(basePath+'/gray/ptrnsValX_AB'+end_path+'.npy',(ptrnsTrainX_A_,ptrnsTrainX_B_))
        #ptrnsTrainX=np.load(basePath+'/gray/ptrnsTrainX.npy');#проверка
        np.save(basePath+'/binary/ptrnsValY_AB'+end_path+'.npy',(ptrnsTrainY_A_,ptrnsTrainY_B_))
   
def double_conv_layer(x,Nc,filters,l2_val,dropout_val,batch_norm):
    if K.image_data_format() == 'channels_last':#определение используемого бэкенда
        axis = 3
    else:
        axis = 1
        
    conv = Conv2D(filters, (3, 3), padding='same',kernel_regularizer=l2(l2_val))(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if Nc>=2:
        conv = Conv2D(filters, (3, 3), padding='same',kernel_regularizer=l2(l2_val))(conv)
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation('relu')(conv)
    if Nc>=3:
        conv = Conv2D(filters, (3, 3), padding='same',kernel_regularizer=l2(l2_val))(conv)
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation('relu')(conv)
     
    if Nc>=4:
        conv = Conv2D(filters, (3, 3), padding='same',kernel_regularizer=l2(l2_val))(conv)
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation('relu')(conv)
    if dropout_val > 0.0:
        conv = SpatialDropout2D(dropout_val)(conv)
    return conv

class my_layer(Layer):
    def __init__(self,por,**kwargs): #инициализация переменной класса и суперкласса
        self.por=por 
        super(my_layer,self).__init__(**kwargs)
        

    def build(self,input_shape):#опредаление настраиваемых перменных
        self.w1 = self.add_weight(name='w1',shape=(input_shape[3],),
        #         initializer=keras.initializers.Constant(value=0.10),trainable=True)
                  initializer=keras.initializers.RandomUniform(minval=0.25,maxval=1.00),trainable=True)
        # self.w2 = self.add_weight(name='w2',shape=(input_shape[3],),
        #           initializer=keras.initializers.RandomUniform(minval=0.5,maxval=1.00),trainable=True)
        # self.por = self.add_weight(name='por',shape=(input_shape[3],),
        #            #initializer=keras.initializers.Constant(value=64.0),trainable=True)   
        #            initializer=keras.initializers.RandomUniform(minval=0.0,maxval=0.5),trainable=True)
        super(my_layer,self).build(input_shape)
        # self.built = True
        
    def call(self,x,y):#проводимые вычисления
        y_=tf.where(y<self.por,0.0,y)
        # y__=tf.where(y_>2*self.por,y_,1)
        #out_put=x*self.w1+y_
        out_put=x*self.w1+y_
        # print(self.w)
        return out_put
    
    def compute_output_shape(self):#определяется форма выхода
         return self.out_put
     
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "por": self.por,#сериализация сохраняет в том числе параметры слоя
        })

def createUNetModel_My_AB(model_name,ptrnShape,Nc,filters,out_size,l2_val,dropout_val,por,batch_norm):
    #Здесь используется функциональная модель API для нелинейных взаимодействия межуд слоями
    #Разница заключается в том, что входной слой для последовательной модели создается и применяется неявно 
    #(и поэтому не будет доступен через атрибут .layers ), 
    #тогда как для моделей, построенных с помощью Functional API, 
    #необходимо явно определить входной слой
    
    if K.image_data_format()  == 'channels_last':
        inputs_a = Input(ptrnShape)
        inputs_b = Input(ptrnShape)
        axis = 3
    else:
        ptrnShape=tuple(reversed(ptrnShape))#перевертывание кортежа
        inputs_a = Input(ptrnShape)
        inputs_b = Input(ptrnShape)
        axis = 1
  
    #outputs_b_1=unet_bin_layer(inputs_b,Nc,filters,l2_val,dropout_val,batch_norm)
      
    #inputs_a_=Conv2D(1, (5, 5),padding='same',activation='linear')(inputs_a)
      #                 kernel_initializer=keras.initializers.Constant(value=0.75),bias_initializer='zeros',
      #                 kernel_regularizer=l2(l2_val))(inputs_a)
      #inputs_a_=LocallyConnected2D(1,(1, 1),padding='same',activation='linear',kernel_regularizer=l2(l2_val))(inputs_a)
    #inputs_b_=Conv2D(1, (5, 5),padding='same',activation='linear')(inputs_b)
      #                 kernel_initializer=keras.initializers.Constant(value=1.00),bias_initializer='zeros',
    x=Add()([inputs_a,inputs_b])
      
    #x = concatenate([inputs_a, inputs_b],axis=axis)
    #x=my_layer(por)(inputs_a,inputs_b)
    conv_1 = double_conv_layer(x,Nc,filters,l2_val,dropout_val,batch_norm)
    down_1 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv_1)
    conv_2 = double_conv_layer(down_1,Nc, 2*filters,l2_val,dropout_val,batch_norm)
    down_2 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_2)

    conv_3 = double_conv_layer(down_2,Nc,4*filters,l2_val,dropout_val,batch_norm)
    down_3 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_3)

    conv_4 = double_conv_layer(down_3,Nc, 8*filters,l2_val,dropout_val,batch_norm)
    down_4 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_4)

    conv_5 = double_conv_layer(down_4,Nc,16*filters,l2_val,dropout_val,batch_norm)
    down_5 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_5)

    conv_6 = double_conv_layer(down_5,Nc,32*filters,l2_val,dropout_val,batch_norm)

    up_1 = UpSampling2D((2,2))(conv_6)
    conv_71 = Conv2D(16*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_1)
    concat_1 = concatenate([conv_5, conv_71], axis=axis)
    conv_7 = double_conv_layer(concat_1,Nc, 16*filters,l2_val,dropout_val,batch_norm)
    

    up_2 = UpSampling2D((2,2))(conv_7)
    conv_81 = Conv2D(8*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_2)
    concat_2 = concatenate([conv_4, conv_81], axis=axis)
    conv_8 = double_conv_layer(concat_2, Nc,8*filters,l2_val,dropout_val,batch_norm)
    
    up_3 = UpSampling2D((2,2))(conv_8)
    conv_91 = Conv2D(4*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_3)
    concat_3 = concatenate([conv_3, conv_91], axis=axis)
    conv_9 = double_conv_layer(concat_3,Nc,4*filters,l2_val,dropout_val,batch_norm)
    
    up_4 = UpSampling2D((2,2))(conv_9)
    conv_101 = Conv2D(2*filters, (2,2), activation='relu',padding='same',kernel_regularizer=l2(l2_val))(up_4)
    concat_4 = concatenate([conv_2, conv_101], axis=axis)
    conv_10 = double_conv_layer(concat_4,Nc,2*filters,l2_val,dropout_val,batch_norm)
    
    up_5 = UpSampling2D((2,2))(conv_10)
    conv_111 = Conv2D(filters, (2,2), activation='relu',padding='same',kernel_regularizer=l2(l2_val))(up_5)
    concat_5 = concatenate([conv_1, conv_111], axis=axis)
    conv_11 = double_conv_layer(concat_5,Nc,filters,l2_val,dropout_val,batch_norm)
    

    if out_size==1:
        conv_12a = Conv2D(out_size, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_11)
        outputs_a = Activation('sigmoid')(conv_12a)
        conv_12b = Conv2D(out_size, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_11)
        outputs_b_2 = Activation('sigmoid')(conv_12b)
    else:
        conv_12a = Conv2D(out_size, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_11)
        outputs_a = Activation('softmax')(conv_12a)
        conv_12b = Conv2D(out_size, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_11)
        outputs_b_2 = Activation('softmax')(conv_12b)
    #outputs_b = concatenate([outputs_b_1,outputs_b_2], axis=axis)       
    #model = Model(inputs_b,conv_10)
    model = Model([inputs_a,inputs_b],[outputs_a,outputs_b_2])

    return model

def average(x, class_weights=None):
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)

def gather_channels(*xs):
    #Преобразование данных в другую форму (разворачивает каналы)
    return xs

def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x
def precision(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, class_weights)
    return score

def recall(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    tp = K.sum(y_true * y_pred, axis=axes)
    fn = K.sum(y_true, axis=axes) - tp
    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, class_weights)
    return score
def f1_score(y_true, y_pred):
    pr = precision(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None)
    re = recall(y_true, y_pred, class_weights=1., smooth=1e-5, threshold=None) 
    f1_score = 2 * (pr * re) / (pr + re)
    return f1_score 
def iou_score(y_true,y_pred,class_weights=1., smooth=1e-5, threshold=None):    
    # y_true = K.one_hot(K.squeeze(K.cast(y_true, tf.int32), axis=-1), n_classes)

    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true + y_pred, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, class_weights)

    return score
def Jaccard_Loss(y_true, y_pred,class_weights=1., smooth=1e-5, threshold=None):
    return 1-iou_score(y_true, y_pred,class_weights=1., smooth=1e-5, threshold=None)
def tversky(y_true, y_pred, alpha=0.7, class_weights=1., smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
    score = average(score,class_weights)
    return score
def tversky_loss(y_true, y_pred, alpha=0.7, class_weights=1., smooth=1e-5, threshold=None):
    return 1-tversky(y_true, y_pred,alpha=0.7, class_weights=1., smooth=1e-5, threshold=None)
def focal_tversky_loss(y_true, y_pred, alpha=0.7, gamma=1.25,lass_weights=1., smooth=1e-5, threshold=None):
    return K.pow(1-tversky(y_true, y_pred,alpha=0.7, class_weights=1., smooth=1e-5, threshold=None),gamma)    
def dice_coef(y_true, y_pred):
   return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
##Определение функции потерь с использованием коэффициента Дайса
def dice_coef_loss(y_true, y_pred):
 	  return -dice_coef(y_true, y_pred)
    
def createUNetModel_My_A(model_name,ptrnShape,Nc,filters,out_size,l2_val,dropout_val,batch_norm):
    #Здесь используется функциональная модель API для нелинейных взаимодействия межуд слоями
    #Разница заключается в том, что входной слой для последовательной модели создается и применяется неявно 
    #(и поэтому не будет доступен через атрибут .layers ), 
    #тогда как для моделей, построенных с помощью Functional API, 
    #необходимо явно определить входной слой
    
    if K.image_data_format()  == 'channels_last':
        inputs_a = Input(ptrnShape)
        axis = 3
    else:
        ptrnShape=tuple(reversed(ptrnShape))#перевертывание кортежа
        inputs_a = Input(ptrnShape)
        axis = 1
    conv_1 = double_conv_layer(inputs_a,Nc,filters,l2_val,dropout_val,batch_norm)
    down_1 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv_1)
    conv_2 = double_conv_layer(down_1,Nc, 2*filters,l2_val,dropout_val,batch_norm)
    down_2 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_2)

    conv_3 = double_conv_layer(down_2,Nc,4*filters,l2_val,dropout_val,batch_norm)
    down_3 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_3)

    conv_4 = double_conv_layer(down_3,Nc, 8*filters,l2_val,dropout_val,batch_norm)
    down_4 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_4)

    conv_5 = double_conv_layer(down_4,Nc,16*filters,l2_val,dropout_val,batch_norm)
    down_5 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_5)

    conv_6 = double_conv_layer(down_5,Nc,32*filters,l2_val,dropout_val,batch_norm)

    up_1 = UpSampling2D((2,2))(conv_6)
    conv_71 = Conv2D(16*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_1)
    concat_1 = concatenate([conv_5, conv_71], axis=axis)
    conv_7 = double_conv_layer(concat_1,Nc, 16*filters,l2_val,dropout_val,batch_norm)
    

    up_2 = UpSampling2D((2,2))(conv_7)
    conv_81 = Conv2D(8*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_2)
    concat_2 = concatenate([conv_4, conv_81], axis=axis)
    conv_8 = double_conv_layer(concat_2, Nc,8*filters,l2_val,dropout_val,batch_norm)
    
    up_3 = UpSampling2D((2,2))(conv_8)
    conv_91 = Conv2D(4*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_3)
    concat_3 = concatenate([conv_3, conv_91], axis=axis)
    conv_9 = double_conv_layer(concat_3,Nc,4*filters,l2_val,dropout_val,batch_norm)
    
    up_4 = UpSampling2D((2,2))(conv_9)
    conv_101 = Conv2D(2*filters, (2,2), activation='relu',padding='same',kernel_regularizer=l2(l2_val))(up_4)
    concat_4 = concatenate([conv_2, conv_101], axis=axis)
    conv_10 = double_conv_layer(concat_4,Nc,2*filters,l2_val,dropout_val,batch_norm)
    
    up_5 = UpSampling2D((2,2))(conv_10)
    conv_111 = Conv2D(filters, (2,2), activation='relu',padding='same',kernel_regularizer=l2(l2_val))(up_5)
    concat_5 = concatenate([conv_1, conv_111], axis=axis)
    conv_11 = double_conv_layer(concat_5,Nc,filters,l2_val,dropout_val,batch_norm)
        
    if out_size==1:
        conv_12 = Conv2D(1, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_11)
        outputs_a = Activation('sigmoid')(conv_12)
    else:
        conv_12b = Conv2D(out_size, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_11)
        outputs_a = Activation('softmax')(conv_12b)
            
    #model = Model(inputs_b,conv_10)
    model = Model(inputs_a,outputs_a)
  
    return model
def createUNetModel_My_All(ptrnShape,Nc,filters,out_size,l2_val,dropout_val,batch_norm):
    #Здесь используется функциональная модель API для нелинейных взаимодействия межуд слоями
    #Разница заключается в том, что входной слой для последовательной модели создается и применяется неявно 
    #(и поэтому не будет доступен через атрибут .layers ), 
    #тогда как для моделей, построенных с помощью Functional API, 
    #необходимо явно определить входной слой
    
    if K.image_data_format()  == 'channels_last':
       inputs_a = Input(ptrnShape)
       inputs_b = Input(ptrnShape)
       axis = 3
    else:
       ptrnShape=tuple(reversed(ptrnShape))#перевертывание кортежа
       inputs_a = Input(ptrnShape)
       inputs_b = Input(ptrnShape)
       axis = 1
        
    conv_1s = double_conv_layer(inputs_a,Nc,filters,l2_val,dropout_val,batch_norm)
    down_1s = MaxPooling2D(pool_size=(2, 2),strides=2)(conv_1s)
    
    conv_1e= double_conv_layer(inputs_b,Nc,filters,l2_val,dropout_val,batch_norm)
    down_1e= MaxPooling2D(pool_size=(2, 2),strides=2)(conv_1e)
    
    down_1=my_layer(por=0.0)(down_1s,down_1e)
    
    
    conv_2s = double_conv_layer(down_1s,Nc,2*filters,l2_val,dropout_val,batch_norm)
    down_2s = MaxPooling2D(pool_size=(2,2),strides=2)(conv_2s)
    
    conv_2e= double_conv_layer(down_1,Nc,2*filters,l2_val,dropout_val,batch_norm)
    down_2e= MaxPooling2D(pool_size=(2, 2),strides=2)(conv_2e)
    
    down_2=my_layer(por=0.0)(down_2s,down_2e)
    
    
    conv_3s = double_conv_layer(down_2s,Nc,4*filters,l2_val,dropout_val,batch_norm)
    down_3s = MaxPooling2D(pool_size=(2,2),strides=2)(conv_3s)
    
    conv_3e= double_conv_layer(down_2,Nc,4*filters,l2_val,dropout_val,batch_norm)
    down_3e= MaxPooling2D(pool_size=(2, 2),strides=2)(conv_3e)
    
    down_3=my_layer(por=0.0)(down_3s,down_3e)

    conv_4s = double_conv_layer(down_3s,Nc, 8*filters,l2_val,dropout_val,batch_norm)
    down_4s = MaxPooling2D(pool_size=(2,2),strides=2)(conv_4s)
    
    conv_4e= double_conv_layer(down_3,Nc,8*filters,l2_val,dropout_val,batch_norm)
    down_4e= MaxPooling2D(pool_size=(2, 2),strides=2)(conv_4e)
    
    down_4=my_layer(por=0.0)(down_4s,down_4e)


    conv_5s = double_conv_layer(down_4s,Nc,16*filters,l2_val,dropout_val,batch_norm)
    down_5s = MaxPooling2D(pool_size=(2,2),strides=2)(conv_5s)
    
    conv_5e= double_conv_layer(down_4,Nc,16*filters,l2_val,dropout_val,batch_norm)
    down_5e= MaxPooling2D(pool_size=(2, 2),strides=2)(conv_5e)
    
    # down_5=my_layer(por=0.0)(down_5s,down_5e)

    # conv_6s = double_conv_layer(down_5s,Nc,32*filters,l2_val,dropout_val,batch_norm)
    # conv_6e = double_conv_layer(down_5,Nc,32*filters,l2_val,dropout_val,batch_norm)
    # concat_6es = concatenate([conv_6s, conv_6e], axis=axis)

    
    concat_5es = concatenate([down_5s, down_5e], axis=axis)

    
    up_1 = UpSampling2D((2,2))(concat_5es)
    conv_71 = Conv2D(16*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_1)
    conv_7 = double_conv_layer(conv_71,Nc, 16*filters,l2_val,dropout_val,batch_norm)
    concat_1 = concatenate([down_4, conv_7], axis=axis)

    up_2 = UpSampling2D((2,2))(concat_1)
    conv_81 = Conv2D(8*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_2)
    conv_8 = double_conv_layer(conv_81, Nc,8*filters,l2_val,dropout_val,batch_norm)
    concat_2 = concatenate([down_3, conv_8], axis=axis)
    
    up_3 = UpSampling2D((2,2))(concat_2)
    conv_91 = Conv2D(4*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_3)
    conv_9 = double_conv_layer(conv_91,Nc,4*filters,l2_val,dropout_val,batch_norm)
    concat_3 = concatenate([down_2, conv_9], axis=axis)
    
    up_4 = UpSampling2D((2,2))(concat_3)
    conv_101 = Conv2D(2*filters, (2,2), activation='relu',padding='same',kernel_regularizer=l2(l2_val))(up_4)
    conv_10 = double_conv_layer(conv_101,Nc,2*filters,l2_val,dropout_val,batch_norm)
    concat_2 = concatenate([down_1, conv_10], axis=axis)
    
    up_5 = UpSampling2D((2,2))(concat_2)
    conv_111 = Conv2D(filters, (2,2), activation='relu',padding='same',kernel_regularizer=l2(l2_val))(up_5)
    conv_11 = double_conv_layer(conv_111,Nc,filters,l2_val,dropout_val,batch_norm)
    #concat_5 = concatenate([down_1, conv_11], axis=axis)    
    
    if out_size==1:
        conv_12 = Conv2D(1, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_11)
        outputs_a = Activation('sigmoid')(conv_12)
    else:
        conv_12b = Conv2D(out_size, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_11)
        outputs_a = Activation('softmax')(conv_12b)
            
    #model = Model(inputs_b,conv_10)
    model = Model([inputs_a,inputs_b],outputs_a)
  
    return model

# Ниже не нужно



    
def createUNetModel_My(model_name,ptrnShape,Nc,filters,out_size,l2_val,dropout_val,batch_norm):
    #Здесь используется функциональная модель API для нелинейных взаимодействия межуд слоями
    #Разница заключается в том, что входной слой для последовательной модели создается и применяется неявно 
    #(и поэтому не будет доступен через атрибут .layers ), 
    #тогда как для моделей, построенных с помощью Functional API, 
    #необходимо явно определить входной слой
    
    if K.image_data_format()  == 'channels_last':
        inputs = Input(ptrnShape)
        axis = 3
    else:
        ptrnShape=tuple(reversed(ptrnShape))#перевертывание кортежа
        inputs = Input(ptrnShape)
        axis = 1
    
   
    if model_name == 'Unet_V_1':
        # inputs2=refine(inputs,por=64.0)
        conv_1 = double_conv_layer(inputs,Nc,filters,l2_val,dropout_val,batch_norm)
        down_1 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv_1)
        conv_2 = double_conv_layer(down_1,Nc, 2*filters,l2_val,dropout_val,batch_norm)
        down_2 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv_2)
    
        conv_3 = double_conv_layer(down_2, Nc,4*filters,l2_val,dropout_val,batch_norm)
        down_3 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv_3)
    
        conv_4 = double_conv_layer(down_3,Nc, 8*filters,l2_val,dropout_val,batch_norm)
        down_4 = MaxPooling2D(pool_size=(2,2),strides=2)(conv_4)
    
        conv_5 = double_conv_layer(down_4,Nc,16*filters,l2_val,dropout_val,batch_norm)
        
        up_1 = UpSampling2D((2,2))(conv_5)
        conv_61 = Conv2D(8*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_1)
        concat_1 = concatenate([conv_4, conv_61], axis=axis)
        conv_6 = double_conv_layer(concat_1,Nc, 8*filters,l2_val,dropout_val,batch_norm)
        
    
        up_2 = UpSampling2D((2,2))(conv_6)
        conv_71 = Conv2D(4*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_2)
        concat_2 = concatenate([conv_3, conv_71], axis=axis)
        conv_7 = double_conv_layer(concat_2, Nc,4*filters,l2_val,dropout_val,batch_norm)
        
        up_3 = UpSampling2D((2,2))(conv_7)
        conv_81 = Conv2D(2*filters, (2,2), activation='relu', padding='same',kernel_regularizer=l2(l2_val))(up_3)
        concat_3 = concatenate([conv_2, conv_81], axis=axis)
        conv_8 = double_conv_layer(concat_3,Nc,2*filters,l2_val,dropout_val,batch_norm)
        
        
        up_4 = UpSampling2D((2,2))(conv_8)
        conv_91 = Conv2D(filters, (2,2), activation='relu',padding='same',kernel_regularizer=l2(l2_val))(up_4)
        concat_4 = concatenate([conv_1, conv_91], axis=axis)
        conv_9 = double_conv_layer(concat_4,Nc,filters,l2_val,dropout_val,batch_norm)
        
        if out_size==1:
            conv_10 = Conv2D(1, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_9)
            conv_10 = Activation('sigmoid')(conv_10)
        else:
            conv_10 = Conv2D(out_size, (1, 1),padding='same',kernel_regularizer=l2(l2_val))(conv_9)
            conv_10 = Activation('softmax')(conv_10)
               
        model = Model(inputs,conv_10)
    
    return model



def testModel_binary(model,model_name,imgName,wsize,out_size,Ap_norm,Ap_clahe,hf): #непосредственно
    if os.path.exists('test_colab/'+model_name+'/')!=True:
        #     os.remove('test_data/'+b)No documentation available 
        os.makedirs('test_colab/'+model_name+'/')
    imgGray = cv2.imread(imgName,0)
    data = np.array(imgGray, dtype='float32')
    data = cv2.GaussianBlur(data,(3,3),1)
    data = cv2.medianBlur(data, 5)
    if hf=='True':
        highpass_1= ndimage.convolve(data, kernel)
        imgGray=data*0.75+highpass_1
        
    elif hf=='Canny':
        imgGray = cv2.Canny(data.astype('uint8'),10,100)
        
    width = imgGray.shape[1]
    height = imgGray.shape[0]
    
    
     # img_=(imgGray/255).astype('float32')
     # ss=0.1*np.random.random((height,width))
     # imgGray=((img_+ss)*255).astype('uint8')
            
    if Ap_clahe==True:
        imgGray = clahe.apply(imgGray)
    if Ap_norm==True:
        imgGray = pred_norm_img(imgGray)
    if Ap_norm==True and Ap_clahe==True:
        imgGray = pred_norm_img(imgGray)
        imgGray = clahe.apply(imgGray)
    
           
    
    
    blocksW = int(width/wsize)
    blocksH = int(height/wsize)
    deltW=0
    deltH=0
    if width % wsize!=0:
        blocksW =blocksW +1
        width_=blocksW*wsize
        deltW=int((width_-width)/blocksW)
        
    if height % wsize!=0:
        blocksH = blocksH+1 #
        height_=blocksH*wsize
        deltH=int((height_- height)/blocksH)
    #step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
         
    imgBin1 = np.zeros((height,width,1), np.uint8)
    imgBin2 = np.zeros((height,width,1), np.uint8)
    imgBin3 = np.zeros((height,width,1), np.uint8)
    for x in range(blocksW):
        # x=1
        if x==0:
            xmin=0; xmax=wsize
        elif x==1:
            xmin=xmax-2*deltW; xmax=xmin+wsize
        else: 
            xmin=xmax-deltW; xmax=xmin+wsize
        for y in range(blocksH):
            # y=1
            if y==0:
                ymin=0; ymax=wsize
            elif y==1:
                ymin=ymax-2*deltH; ymax=ymin+wsize
            else:
                ymin=ymax-deltH; ymax=ymin+wsize
            
            imgRoiGray = imgGray[ymin : ymax, xmin : xmax]
            imgRoiGray = imgRoiGray.astype('float32') / 255.0
    
            ptrnTestX = []
            ptrnTestX.append(imgRoiGray)
            ptrnTestX = np.array(ptrnTestX)
            ptrnTestX_ = ptrnTestX.reshape(ptrnTestX.shape[0], wsize, wsize, 1)
                        
            
            prediction_ = model.predict(ptrnTestX_)
            prediction = (prediction_*255).astype('int8')
            prediction = prediction.reshape(wsize, wsize, out_size)
                  
            
            if out_size==1:
                #внимане второй индекс уменьшается на единицу
                imgBin1[ymin:ymax,xmin:xmax] = prediction
                imgBin2[ymin:ymax,xmin:xmax] = prediction
            elif out_size==2:
                imgBin1[ymin:ymax,xmin:xmax,0] = prediction[:, :, 0]
                imgBin2[ymin:ymax,xmin:xmax,0] = prediction[:, :, 1]
                for col in range(wsize):
                    for row in range(wsize):
                        imgBin2[ymin + row, xmin + col,0] = prediction[row, col, 1]
                        if prediction[row, col,1]<prediction[row, col, 0]:
                            imgBin3[ymin + row, xmin + col,0]=255
    print('imgGray')
    a=imgName[11:len(imgName)-4] #для переименования файла
    b=a+'.jpg'
    cv2.imwrite('test_colab/'+model_name+'/'+b,imgGray)
    print('imgBin')
    b=a+'_3.png'
    cv2.imwrite('test_colab/'+model_name+'/'+b,imgBin3)
      
        
    return (imgBin1,imgBin2,imgBin3,prediction)
    #конец функции

#Опредление функции для загрузки данных из папки с заданным путем basePath
#и формирования тестирующей выборки с нарезкой
def loadTestData_binary_colab(basePath,wsize,Ap_norm,Ap_clahe,hf):
    #out_size==2
    ptrnsTestX = []
    ptrnsTestY = []
   
    path_gray = basePath +'/gray 2014/1me'
    path_BIN=basePath +'/gray 2014/train_colab_bin'
    
    k=0;
    #os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри 
    for fileName in os.listdir(path_gray):#дает все что есть в каталоге
        k=k+1;
        if k % 5 !=0: #остаток от деления
            imgGray = cv2.imread(path_gray + '/' + fileName, 0)
            fileName_=fileName[0:len(fileName)-4]+'_3.jpg'
            imgBIN =  cv2.imread(path_BIN + '/' + fileName_, 0)
            # width = imgGray.shape[1]
            # height = imgGray.shape[0]
            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((2000,2000))
            # imgGray=((img_+ss)*255).astype('uint8')
                     
            if Ap_clahe==True:
                imgGray = clahe.apply(imgGray)
            if Ap_norm==True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm==True and Ap_clahe==True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print('name ', fileName)
            print('name ', fileName_)
            
            # createPtrns_Gray_2D(imgGray,imgBIN, wsize, ptrnsTestX)
            # imgGray_=imgGray.T
            # imgBIN_=imgBIN.T
            # createPtrns_Gray_2D(imgGray_,imgBIN_, wsize, ptrnsTestX)
            createPtrns_Gray(imgGray, wsize, ptrnsTestX,hf)
            imgGray_=imgGray.T
            createPtrns_Gray(imgGray_, wsize, ptrnsTestX,hf)
    path = basePath + "/binary 2014/1me"
    k=0;
    for fileName in os.listdir(path):
        k=k+1
        if k % 5 !=0:
            imgBin = cv2.imread(path + '/' + fileName, 0)
            print('name ', fileName)
            createPtrns_Bin(imgBin, wsize, ptrnsTestY,False)
            imgBin_=imgBin.T
            createPtrns_Bin(imgBin_, wsize, ptrnsTestY,False)

    ptrnsTestX = np.array(ptrnsTestX)
    #развертка в матрицу изображения
    ptrnsTestX = ptrnsTestX.reshape(ptrnsTestX.shape[0], wsize, wsize, 1) #3
    
        
    #развертка в матрицу изображения
    ptrnsTestY = np.array(ptrnsTestY)
    ptrnsTestY = ptrnsTestY.reshape(ptrnsTestY.shape[0], wsize, wsize,1) #3
    return (ptrnsTestX,ptrnsTestY)
#Создание выборки для валидации
def testModel_binary_val(model,model_name,basePath,wsize,out_size,Ap_norm,Ap_clahe,hf): #непосредственно
    
    path_gray = basePath +'/gray 2014/1me'
    path_BIN=basePath +'/gray 2014/train_colab_bin'
    if os.path.exists('test_colab/'+model_name+'/')!=True:
        #     os.remove('test_data/'+b)No documentation available 
        os.makedirs('test_colab/'+model_name+'/')
    k=0     
    for fileName in os.listdir(path_gray):#дает все что есть в каталоге
        k=k+1;
        # if (k-1) % 10 ==0: #остаток от деления
        if (k-1) % 5 ==0: #остаток от деления
            imgGray = cv2.imread(path_gray + '/' + fileName, 0)
            fileName_=fileName[0:len(fileName)-4]+'_3.jpg'
            imgBIN =  cv2.imread(path_BIN + '/' + fileName_,0)
            
            data = np.array(imgGray, dtype='float32')
            data = cv2.GaussianBlur(data,(3,3),1)
            data = cv2.medianBlur(data, 5)
            if hf=='True':
                highpass_1= ndimage.convolve(data, kernel)
                imgGray=data*0.75+highpass_1
                
            elif hf=='Canny':
                imgGray = cv2.Canny(data.astype('uint8'),10,100)
            
            # data = np.array(imgGray, dtype='float32')
            # data = cv2.GaussianBlur(data,(5,5),cv2.BORDER_DEFAULT)
            # imgGray=data+imgBIN.astype('float32')
            
            
            width = imgGray.shape[1]
            height = imgGray.shape[0]
            
            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((2000,2000))
            # imgGray=((img_+ss)*255).astype('uint8')
            
            if Ap_clahe==True:
                imgGray = clahe.apply(imgGray)
            if Ap_norm==True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm==True and Ap_clahe==True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print('name ', fileName)
               
            
            blocksW = int(width/wsize)
            blocksH = int(height/wsize)
            deltW=0
            deltH=0
            if width % wsize!=0:
                blocksW =blocksW +1
                width_=blocksW*wsize
                deltW=int((width_-width)/blocksW)
            if height % wsize!=0:
                blocksH = blocksH+1 #
                height_=blocksH*wsize
                deltH=int((height_- height)/blocksH)
                                    
            imgBin3 = np.zeros((height,width,1), np.uint8)
            for x in range(blocksW):
                if x==0:
                    xmin=0; xmax=wsize
                elif x==1:
                    xmin=xmax-2*deltW; xmax=xmin+wsize
                else: 
                    xmin=xmax-deltW; xmax=xmin+wsize
                for y in range(blocksH):
                    # y=1
                    if y==0:
                        ymin=0; ymax=wsize
                    elif y==1:
                        ymin=ymax-2*deltH; ymax=ymin+wsize
                    else:
                        ymin=ymax-deltH; ymax=ymin+wsize
                    
                    imgRoiGray = imgGray[ymin : ymax, xmin : xmax]
                    imgRoiGray = imgRoiGray.astype('float32') / 255.0
            
                    ptrnTestX = []
                    ptrnTestX.append(imgRoiGray)
                    ptrnTestX = np.array(ptrnTestX)
                    ptrnTestX_ = ptrnTestX.reshape(ptrnTestX.shape[0], wsize, wsize, 1)
                                    
                    prediction_ = model.predict(ptrnTestX_)
                    prediction = (prediction_*255).astype('int8')
                    prediction = prediction.reshape(wsize, wsize, out_size)
                          
                
                    if out_size==1:
                        #внимане второй индекс уменьшается на единицу
                        imgBin3[ymin:ymax,xmin:xmax,0] = prediction
                    elif out_size==2:
                         for col in range(wsize):
                            for row in range(wsize):
                                if prediction[row, col,1]<prediction[row, col, 0]:
                                    imgBin3[ymin + row, xmin + col,0]=255
            print('imgGray')
            a=fileName[0:len(fileName)-4] #для переименования файла
            b=a+'.jpg'
            cv2.imwrite('test_colab/'+model_name+'/'+b,imgGray)
            print('imgBin')
            b=a+'_3.png'
            cv2.imwrite('test_colab/'+model_name+'/'+b,imgBin3)
    return imgBin3
def createPtrns_Bin(img, wsize,ptrnTrain,hf):
    if hf !=False:
        data = np.array(img, dtype='float32')
        data = cv2.GaussianBlur(data,(3,3),1)
        data = cv2.medianBlur(data, 5)
        img= ndimage.convolve(data, kernel)
        nmi=np.where(img<128)
        img[nmi]=0
        nma=np.where(img>=128)
        img[nma]=255
    width = img.shape[1] #размеры входного изображения (по X)
    height = img.shape[0] #размеры входного изображения (по Y)
    blocksW = int(width/wsize) #число блоков
    blocksH = int(height/wsize)
    deltW=0
    deltH=0
    if width % wsize!=0:
        blocksW =blocksW +1
        width_=blocksW*wsize
        deltW=int((width_-width)/blocksW)
        
    if height % wsize!=0:
        blocksH = blocksH+1 #
        height_=blocksH*wsize
        deltH=int((height_- height)/blocksH)
    #step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
    for x in range(blocksW): #*2):
        if x==0:
            xmin=0; xmax=wsize
        elif x==1:
            xmin=xmax-2*deltW; xmax=xmin+wsize
        else: 
            xmin=xmax-deltW; xmax=xmin+wsize
        # print(xmin,xmax)
        for y in range(blocksH): #*2):
            if y==0:
                ymin=0; ymax=wsize
            elif y==1:
                ymin=ymax-2*deltH; ymax=ymin+wsize
            else:
                ymin=ymax-deltH; ymax=ymin+wsize
            # print(ymin,ymax)
                
            imgRoi = img[ymin : ymax, xmin : xmax]
            imgRoi = imgRoi.astype('float32') / 255.0
            ptrnTrain.append(imgRoi) #добавляет объект в конце списка
def createPtrns_Gray_2D(img,img_bin, wsize,ptrnTrain):
    data = np.array(img, dtype='float32')
    data = cv2.GaussianBlur(data,(5,5),cv2.BORDER_DEFAULT)
    img=data+img_bin.astype('float32')
    width = img.shape[1] #размеры входного изображения (по X)
    height = img.shape[0] #размеры входного изображения (по Y)
    imgRoi=np.zeros((wsize,wsize),np.float32)
    
    blocksW = int(width/wsize) #число блоков
    blocksH = int(height/wsize)
    deltW=0
    deltH=0
    if width % wsize!=0:
        blocksW =blocksW +1
        width_=blocksW*wsize
        deltW=int((width_-width)/blocksW)
        
    if height % wsize!=0:
        blocksH = blocksH+1 #
        height_=blocksH*wsize
        deltH=int((height_- height)/blocksH)
    
    for x in range(blocksW): #*2):
        if x==0:
            xmin=0; xmax=wsize
        elif x==1:
            xmin=xmax-2*deltW; xmax=xmin+wsize
        else: 
            xmin=xmax-deltW; xmax=xmin+wsize
        # print(xmin,xmax)
        for y in range(blocksH): #*2):
            if y==0:
                ymin=0; ymax=wsize
            elif y==1:
                ymin=ymax-2*deltH; ymax=ymin+wsize
            else:
                ymin=ymax-deltH; ymax=ymin+wsize
            # print(ymin,ymax)
                
            imgRoi = img[ymin : ymax, xmin : xmax]
            # imgRoi[:,:,1] = img_hf[ymin : ymax, xmin : xmax]
            imgRoi = imgRoi/255.0
            ptrnTrain.append(imgRoi) #добавляет объект в конце списка
#Опредление функции для загрузки данных из папки с заданным путем basePath
#и формирования обучающей выборки с нарезкой
def loadTrainData_binary_colab(basePath,wsize,Ap_norm,Ap_clahe,hf):
    #out_size==2
    ptrnsTrainX = []
    ptrnsTrainY = []
    #смешанные данные
    # path_gray = basePath + "/gray_mixt"
    # path_binary = basePath + "/binary_mixt"
    #данные из одного слоя
    # path_gray = basePath +'/gray 2014/1me'
    path_gray = basePath +'/gray 2014/1me'
    path_BIN=basePath +'/gray 2014/train_colab_bin'
    path_binary = basePath + '/binary 2014/1me'
    k=0
    l=0
    #os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри 
    for fileName in os.listdir(path_gray):#дает все что есть в каталоге
        k=k+1;
        if k % 5==0: #остаток от деления
            l=l+1
            imgGray = cv2.imread(path_gray + '/' + fileName, 0)
            fileName_=fileName[0:len(fileName)-4]+'_3.jpg'
            imgBIN =  cv2.imread(path_BIN + '/' + fileName_, 0)
            width = imgGray.shape[1] #размеры входного изображения (по X)
            height = imgGray.shape[0] 
            
            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((height,width))
            # imgGray=((img_+ss)*255).astype('uint8')
            
            if Ap_clahe==True and Ap_norm==False:
                imgGray = clahe.apply(imgGray)
            if Ap_clahe==False and Ap_norm==True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm==True and Ap_clahe==True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print('name ', fileName)
            print('name ', fileName_)
            
            # createPtrns_Gray_2D(imgGray,imgBIN, wsize, ptrnsTrainX)
            # imgGray_=imgGray.T
            # imgBIN_=imgBIN.T
            # createPtrns_Gray_2D(imgGray_,imgBIN_,wsize, ptrnsTrainX)
            
            createPtrns_Gray(imgGray, wsize, ptrnsTrainX,hf)
            imgGray_=imgGray.T
            createPtrns_Gray(imgGray_, wsize, ptrnsTrainX,hf)
    print(l)
    
    k=0
    l=0
    for fileName in os.listdir(path_binary):
        k=k+1
        if k % 5==0:
            l=l+1
            imgBin = cv2.imread(path_binary + '/' + fileName, 0)
            print('name ', fileName)
            createPtrns_Bin(imgBin, wsize, ptrnsTrainY,False)
            imgBin_=imgBin.T
            createPtrns_Bin(imgBin_, wsize, ptrnsTrainY,False)
    print(l)

    ptrnsTrainX = np.array(ptrnsTrainX)
    #развертка в матрицу изображения
    ptrnsTrainX = ptrnsTrainX.reshape(ptrnsTrainX.shape[0], wsize, wsize, 1) #3
    
        
    #развертка в матрицу изображения
    ptrnsTrainY = np.array(ptrnsTrainY)
    ptrnsTrainY = ptrnsTrainY.reshape(ptrnsTrainY.shape[0], wsize, wsize,1) #3
    return (ptrnsTrainX,ptrnsTrainY)
#Создание выборки для валидации
def loadValData_binary_colab(basePath,wsize,Ap_norm,Ap_clahe,hf):
    ptrnsValX = []
    ptrnsValY = []
    #смешанные данные
    # path_gray = basePath + "/gray_mixt"
    # path_binary = basePath + "/binary_mixt"
    #данные из одного слоя
    # path_gray = basePath +'/gray 2014/1me'
    path_gray = basePath +'/gray 2014/1me'
    path_BIN=basePath +'/gray 2014/train_colab_bin'
    path_binary = basePath + '/binary 2014/1me'
    k=0
    l=0
    #os.makedirs(os.path.join('данные', "bad")) #функция создания подпапки внутри 
    for fileName in os.listdir(path_gray):#дает все что есть в каталоге
        k=k+1
        if (k-1) % 10 ==0:
            l=l+1
            imgGray = cv2.imread(path_gray + '/' + fileName, 0)
            fileName_=fileName[0:len(fileName)-4]+'_3.jpg'
            imgBIN =  cv2.imread(path_BIN + '/' + fileName_, 0)
            width = imgGray.shape[1] #размеры входного изображения (по X)
            height = imgGray.shape[0] 
            # img_=(imgGray/255).astype('float32')
            # ss=0.1*np.random.random((height,width))
            # imgGray=((img_+ss)*255).astype('uint8')
            
            if Ap_clahe==True and Ap_norm==False:
                imgGray = clahe.apply(imgGray)
            if Ap_clahe==False and Ap_norm==True:
                imgGray = pred_norm_img(imgGray)
            if Ap_norm==True and Ap_clahe==True:
                imgGray = pred_norm_img(imgGray)
                imgGray = clahe.apply(imgGray)
            print('name ', fileName)
            print('name ', fileName_)
            
            # createPtrns_Gray_2D(imgGray,imgBIN, wsize, ptrnsValX)
            # imgGray_=imgGray.T
            # imgBIN_=imgBIN.T
            # createPtrns_Gray_2D(imgGray_,imgBIN_, wsize, ptrnsValX)
            
            createPtrns_Gray(imgGray, wsize, ptrnsValX,hf)
            imgGray_=imgGray.T
            createPtrns_Gray(imgGray_, wsize, ptrnsValX,hf)
    print(l)
    
    k=0
    l=0
    for fileName in os.listdir(path_binary):
        k=k+1
        if (k-1) % 10==0:
            l=l+1
            imgBin = cv2.imread(path_binary + '/' + fileName, 0)
            print('name ', fileName)
            createPtrns_Bin(imgBin,wsize,ptrnsValY,False)
            imgBin_=imgBin.T
            createPtrns_Bin(imgBin_, wsize, ptrnsValY,False)
    print(l)    
    ptrnsValX = np.array(ptrnsValX)
    #развертка в матрицу изображения
    ptrnsValX = ptrnsValX.reshape(ptrnsValX.shape[0], wsize, wsize, 1) #3
    #развертка в матрицу изображения
    ptrnsValY = np.array(ptrnsValY)
    ptrnsValY = ptrnsValY.reshape(ptrnsValY.shape[0], wsize, wsize, 1) #3
    return (ptrnsValX,ptrnsValY)
def shuffle_unson_write(basePath,ptrnsTrainX,ptrnsTrainY,wsize,out_size,Ap_norm,Ap_clahe):
    #Перемешивание данных для последдующей записи
    
    a=ptrnsTrainX.shape
    b=ptrnsTrainY.shape
    assert a[0] == b[0]
    np.random.seed(10)
    p = np.random.permutation(a[0])
    
    ptrnsTrainX_=ptrnsTrainX[p,:,:,:]
    ptrnsTrainY_=ptrnsTrainY[p,:,:,:]
    end_path='_single_'+str(wsize)+'_'+str(out_size)+'_'+str(Ap_norm)+'_'+str(Ap_clahe)
    if basePath=='train_colab':
        np.save(basePath+'/gray/ptrnsTrainX'+end_path+'.npy',ptrnsTrainX_)
        #ptrnsTrainX=np.load(basePath+'/gray/ptrnsTrainX.npy');#проверка
        np.save(basePath+'/binary/ptrnsTrainY'+end_path+'.npy',ptrnsTrainY_)
    elif basePath=='val_colab':
        np.save(basePath+'/gray/ptrnsValX'+end_path+'.npy',ptrnsTrainX_)
        #ptrnsTrainX=np.load(basePath+'/gray/ptrnsTrainX.npy');#проверка
        np.save(basePath+'/binary/ptrnsValY'+end_path+'.npy',ptrnsTrainY_)
      
    #ptrnsTrainY=np.load(basePath+'/binary'/ptrnsTrainY.npy);#проверк
    #return ptrnsTrainX_,ptrnsTrainY_
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)
##Определение функции потерь с использованием коэффициента Дайса
def dice_coef_loss(y_true, y_pred):
 	return -dice_coef(y_true, y_pred)
def createPtrns_Bin(img, wsize,ptrnTrain,hf):
    if hf !=False:
        data = np.array(img, dtype='float32')
        data = cv2.GaussianBlur(data,(3,3),1)
        data = cv2.medianBlur(data, 5)
        img= ndimage.convolve(data, kernel)
        nmi=np.where(img<128)
        img[nmi]=0
        nma=np.where(img>=128)
        img[nma]=255
    width = img.shape[1] #размеры входного изображения (по X)
    height = img.shape[0] #размеры входного изображения (по Y)
    blocksW = int(width/wsize) #число блоков
    blocksH = int(height/wsize)
    deltW=0
    deltH=0
    if width % wsize!=0:
        blocksW =blocksW +1
        width_=blocksW*wsize
        deltW=int((width_-width)/blocksW)
        
    if height % wsize!=0:
        blocksH = blocksH+1 #
        height_=blocksH*wsize
        deltH=int((height_- height)/blocksH)
    #step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
    for x in range(blocksW): #*2):
        if x==0:
            xmin=0; xmax=wsize
        elif x==1:
            xmin=xmax-2*deltW; xmax=xmin+wsize
        else: 
            xmin=xmax-deltW; xmax=xmin+wsize
        # print(xmin,xmax)
        for y in range(blocksH): #*2):
            if y==0:
                ymin=0; ymax=wsize
            elif y==1:
                ymin=ymax-2*deltH; ymax=ymin+wsize
            else:
                ymin=ymax-deltH; ymax=ymin+wsize
            # print(ymin,ymax)
                
            imgRoi = img[ymin : ymax, xmin : xmax]
            imgRoi = imgRoi.astype('float32') / 255.0
            ptrnTrain.append(imgRoi) #добавляет объект в конце списка
#Определение функции для нарезания блоков размера wsize во входном изображении и создания паттернов
def createPtrns_Gray(img, wsize,ptrnTrain,hf):
    data = np.array(img, dtype='float32')
    data = cv2.GaussianBlur(data,(3,3),1)
    data = cv2.medianBlur(data, 5)
    if hf=='True':
        highpass_1= ndimage.convolve(data, kernel)
        img=data*0.75+highpass_1
    elif hf=='Canny':
        img = cv2.Canny(data.astype('uint8'),10,100)
    width = img.shape[1] #размеры входного изображения (по X)
    height = img.shape[0] #размеры входного изображения (по Y)
    blocksW = int(width/wsize) #число блоков
    blocksH = int(height/wsize)
    deltW=0
    deltH=0
    if width % wsize!=0:
        blocksW =blocksW +1
        width_=blocksW*wsize
        deltW=int((width_-width)/blocksW)
        
    if height % wsize!=0:
        blocksH = blocksH+1 #
        height_=blocksH*wsize
        deltH=int((height_- height)/blocksH)
    #step = int(wsize/2) #шаг нарезки можно уменьшить для перекрытия блоков
    for x in range(blocksW): #*2):
        if x==0:
            xmin=0; xmax=wsize
        elif x==1:
            xmin=xmax-2*deltW; xmax=xmin+wsize
        else: 
            xmin=xmax-deltW; xmax=xmin+wsize
        # print(xmin,xmax)
        for y in range(blocksH): #*2):
            if y==0:
                ymin=0; ymax=wsize
            elif y==1:
                ymin=ymax-2*deltH; ymax=ymin+wsize
            else:
                ymin=ymax-deltH; ymax=ymin+wsize
            # print(ymin,ymax)
                
            imgRoi = img[ymin : ymax, xmin : xmax]
            imgRoi = imgRoi.astype('float32') / 255.0
            ptrnTrain.append(imgRoi) #добавляет объект в конце списка
            #Определение функции для нарезания блоков размера wsize во входном изображении и создания паттернов
