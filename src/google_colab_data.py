#Файл для формирования данных для google colab 
#и тестирования сети из google colab
#Импорт всех необходимых компонентов и утилит
#Импорт всех необходимых компонентов и утилит
import cv2 #это opencv
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import keras
from keras import backend as K #Для программирования собственных метрик
import tensorflow as tf
from keras import layers
from tensorflow.keras.utils import plot_model
from keras.layers import Conv2D,Add,Activation, Dropout,Dense
from keras.layers import Input,MaxPooling2D,UpSampling2D,concatenate
from keras.models import Sequential, load_model, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D
from tensorflow.python.keras import layers
from keras import optimizers
from keras import regularizers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
# from keras.utils import plot_model
# from tensorflow.keras.utils import plot_model
from tensorflow.python.keras import models
from keras.optimizers import Adam,RMSprop
from cv2 import imshow
from cv2  import CLAHE
from keras.models import load_model,Model
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import get_custom_objects
import random
from evaluate import testModel_binary, testModel_binary_AB, testModel_binary_val
from load import loadTestData_binary_colab_AB, loadTrainData_binary_colab, loadTrainData_binary_colab_AB, loadValData_binary_colab_AB
from make_patches import createPtrns_Bin_AB, createPtrns_Gray_AB
from metrics import f1_score, tversky, tversky_loss
from models import createUNetModel_My, createUNetModel_My_A, createUNetModel_My_AB, double_conv_layer, my_layer
from load import loadValData_binary_colab
from image_proccess import pred_norm_img
from my_fun_new_colab import shuffle_unson_write_AB
from models import createUNetModel_My_All
from my_fun_new_colab import dice_coef,dice_coef_loss
from evaluate import testModel_binary_val_AB
#from google.colab.patches import cv2_imshow
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(32,32))#адападаптивная эквализация


random.seed(10)
model_name='Unet_V_3'
out_size=2  #количество выходов сетиdocumentation a
dropout_val=0.0
l2_val=0.00001
filters = 16 #число фильтров свертки*X для 128 32 для 256?????
Nc=1 #число всерток в стандартном блоке
Ap_norm=False
Ap_clahe=False
B_norm=True
hf='True'
por=0.20
#por_='0.00'
w='AB_sum_0.50_rms_sc'
print(dropout_val,filters,l2_val,out_size,Ap_norm,Ap_clahe)
wsize = 512
ptrnShape = (wsize, wsize, 1)
tp_files='single'
end_path='_'+tp_files+'_'+str(wsize)+'_'+str(Ap_norm)+'_'+str(Ap_clahe)+'_'+str(por)
e='_'+str(out_size)+'_'+str(Nc)+'_'+str(filters)+'_'+str(dropout_val)+'_'+str(w)+'_'+str(B_norm)
end_path_class=end_path+e
model_name=model_name+end_path_class+'.h5'


# Создание паттернов из всех изображений папки 
basePath='train_colab'
(ptrnsTrainX_A,ptrnsTrainX_B,ptrnsTrainY_A,ptrnsTrainY_B) = loadTrainData_binary_colab_AB(basePath,wsize,Ap_norm,Ap_clahe,hf,por)
(ptrnsValX_A,ptrnsValX_B,ptrnsValY_A,ptrnsValY_B) = loadValData_binary_colab_AB(basePath,wsize,Ap_norm,Ap_clahe,hf,por)
#перемешивание данных и запись в файл
basePath='train_colab'
shuffle_unson_write_AB(basePath,end_path,ptrnsTrainX_A,ptrnsTrainX_B,ptrnsTrainY_A,ptrnsTrainY_B)
basePath='val_colab'
shuffle_unson_write_AB(basePath,end_path,ptrnsValX_A,ptrnsValX_B,ptrnsValY_A,ptrnsValY_B)





# basePath='val_colab'
# #Сохранение парметров предобработки
# np.save(basePath+'/pred_param'+end_path+'.npy',pred_param);

# # # # # #Создание модели сети
#@title Текст заголовка по умолчанию
#model.summary()

# model_name = 'Unet_V_All'+end_path_class+'.h5'
# get_custom_objects().update({'tversky_loss':tversky_loss})
# model = createUNetModel_My_All(ptrnShape,Nc,filters,out_size,l2_val,dropout_val,B_norm)
# # model.summary()
# # # # # # opt=AdaNo dom(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)#,decay=1e-6)
# opt=RMSprop(learning_rate=0.0005) #это пока лучший вариант
# model.compile(opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #model.compile(opt,loss='tversky_loss', metrics=['accuracy'])

# filepath='F:/Segmentation/'+model_name
# checkpointer = ModelCheckpoint(filepath,monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False)
# reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',verbose=1,mode='auto',factor=0.75, patience=5, min_lr=0.0001)

# # X=[ptrnsTrainX_A,ptrnsTrainX_B];Y=[ptrnsTrainY_A,ptrnsTrainY_B]; VX=[ptrnsValX_A,ptrnsValX_B];VY=[ptrnsValY_A,ptrnsValY_B]
# # X=ptrnsTrainX_A;Y=ptrnsTrainY_A;VX=ptrnsValX_A;VY=ptrnsValY_A
# X=[ptrnsTrainX_A,ptrnsTrainX_B];Y=ptrnsTrainY_A; VX=[ptrnsValX_A,ptrnsValX_B];VY=ptrnsValY_A
# history = model.fit(X,Y,batch_size=4, epochs=32,verbose=1,validation_data=(VX,VY),shuffle=True)#,callbacks=[checkpointer,reduce_lr])


# #История accuracy
# model.save(filepath)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend(['train','validation'], loc='upper left')
# plt.grid()
# plt.show()
 
# #loss история
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.grid()
# plt.show()

#Загрузка ранее сохраненного
basePath='train_colab'
(ptrnsTrainX_A,ptrnsTrainX_B)=np.load(basePath+'/gray/ptrnsTrainX_AB'+end_path+'.npy');#проверка
(ptrnsTrainY_A,ptrnsTrainY_B)=np.load(basePath+'/binary/ptrnsTrainY_AB'+end_path+'.npy');#проверка
basePath='val_colab'
(ptrnsvalX_A,ptrnsvalX_B)=np.load(basePath+'/gray/ptrnsValX_AB'+end_path+'.npy');#проверка
(ptrnsValY_A,ptrnsValY_B)=np.load(basePath+'/binary/ptrnsValY_AB'+end_path+'.npy');#проверка

# #Тестирование
# pred_param__=np.load(basePath+'/pred_param'+end_path+'.npy');
# pred_param=pred_param__.tolist()
get_custom_objects().update({'tversky_loss':tversky_loss,'tversky':tversky})
get_custom_objects().update({'f1_score':f1_score})




model = keras.models.load_model(model_name,custom_objects={'my_layer': my_layer})
print(model.layers[0].weights)
print(model.layers[2].weights)

#model.layers[2].por
# imgName='train_colab/gray 2014/1me/sl_II_m1_0030.jpg'
# imgName='train_colab/gray 2014/1me/sl_II_m1_0013.jpg'
# imgName='train_data/sl_II_m1_0056.jpg'
#imgName='train_colab/gray 2014/1me/sl_II_m1_0076.jpg'
#model.summary()

imgName='train_data/94.jpg'
# imgName='train_data/SL_II_M2_0055.jpg'
# imgName='train_data/sl_II_m1_0076.jpg'
(img_A,img_B,imgBin3A,imgBin3B,prediction)=testModel_binary_AB(model,model_name,imgName,wsize,out_size,Ap_norm,Ap_clahe,hf,por)
a=prediction[0,]
basePath='train_colab'
#визуальное тестирование в цикле валидацилнных данных
imgBin3=testModel_binary_val_AB(model,model_name,basePath,wsize,out_size,Ap_norm,Ap_clahe,hf,por)
#Тестирование по ошибке

(ptrnsTestX_A,ptrnsTestX_B,ptrnsTestY_A,ptrnsTestY_B) = loadTestData_binary_colab_AB(basePath,wsize,Ap_norm,Ap_clahe,hf,por)
fix=model_name[0:8] 
if fix=='Unet_V_0':
    result = model.evaluate(ptrnsTestX_A,ptrnsTestY_A,verbose=1, batch_size=1)
elif fix=='Unet_V_1'or fix=='Unet_V_4':
    result = model.evaluate([ptrnsTestX_A,ptrnsTestX_B],ptrnsTestY_A,verbose=1, batch_size=1)
else:
    result = model.evaluate([ptrnsTestX_A,ptrnsTestX_B],[ptrnsTestY_A,ptrnsTestY_B],verbose=1, batch_size=1)
#loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)
Ac_sum=result[1]*100;
print(Ac_sum)
# d=ptrnsTestX.shape
# results=np.zeros((d[0],2),np.float32)
# for i in range(d[0]):
#     x=ptrnsTestX[i,:,:,0]
#     x = x.reshape(1, wsize, wsize, 1)
#     y=ptrnsTestY[i,:,:,0]
#     y = y.reshape(1, wsize, wsize, 1)
#     results[i,:]=model.evaluate(x,y,verbose=0, batch_size=1)
#     #print(results[i,:])
# r=np.mean(results) #результат аналогиный