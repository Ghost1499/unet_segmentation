#Импорт всех необходимых компонентов и утилит
import cv2 #это opencv
import os
import numpy as np
import keras
from keras import backend as K #Для программирования собственных метрик
import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Conv2D,Activation,Dropout,Dense
from keras.layers import Input,MaxPooling2D,UpSampling2D,concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras.models import Sequential, load_model, Model
from keras import layers
from keras import optimizers
from keras import regularizers
from keras.regularizers import l2
from keras import models
from keras.optimizers import Adam
from cv2 import imshow
from cv2  import CLAHE
from matplotlib import pyplot as plt
from src.my_fun_new_colab import pred_norm,loadTrainData,loadValData,createPatternsGrayBin_num_class
from my_fun1 import loadTrainData_binary,loadValData_binary,createPtrns,testModel_binary
from my_fun1 import dice_coef,dice_coef_loss,jacard_coef,jacard_coef_loss
from my_fun1 import double_conv_layer,createUNetModel,createUNetModel_My,createUNetModel_dconv,testModel
from keras import utils
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import random
#from google.colab.patches import cv2_imshow
clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(32,32))#адаптивная эквализация
out_size= 2  #количество выходов сети
dropout_val=0.0
l2_val=0.00001
filters = 32 #число фильтров свертки*X для 128 32 для 256?????
Ap_norm=False
Ap_clahe=True
print(dropout_val,filters,l2_val,out_size,Ap_norm,Ap_clahe)
wsize = 128
ptrnShape = (wsize, wsize, 1)
modelName = 'classifier.h5'
random.seed(10);#задание фиксированного старта для случайных величин

# Создание паттернов из всех изображений папки 
basePath='train_data'
if out_size==3:
    (ptrnsTrainX,ptrnsTrainY,pred_param) = loadTrainData(basePath,wsize,Ap_norm,Ap_clahe)
else:
    (ptrnsTrainX,ptrnsTrainY,pred_param) = loadTrainData_binary(basePath,wsize,Ap_norm,Ap_clahe)
basePath='val_data'
if out_size==3:
    (ptrnsValX, ptrnsValY) = loadValData(basePath,wsize,pred_param,Ap_clahe)
else:
    (ptrnsValX, ptrnsValY) = loadValData_binary(basePath,wsize,pred_param,Ap_clahe)
# # Преобразуем метки в категории
# ptrnsTrainY = np_utils.to_categorical(ptrnsTrainY,num_classes=3)
# ptrnsValY = np_utils.to_categorical(ptrnsValY,num_classes= 3)


#Создание модели сети
model = createUNetModel(ptrnShape)
#model = createUNetModel_My(ptrnShape,filters,out_size,l2_val,dropout_val)
#model=createUNetModel_dconv(ptrnShape,filters,out_size,l2_val,dropout_val)
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
#'это прамтры по умиолчанию
opt=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)#,decay=1e-6)

if out_size==1:
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
elif out_size==2:
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
else:
    model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])
#вызов оптимизатора по имени
# model.compile('adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.compile(optimizer=Adam(lr=1e-4),loss= dice_coef_loss,metrics=[dice_coef])
#model.compile(optimizer='adam',loss= jaccard_loss,metrics=[jaccard])

# model.summary()
checkpointer = ModelCheckpoint(filepath='tmp_model_/model_best_'+str(wsize)+'_'+str(out_size)+'_.h5',monitor='val_accuracy', 
                 verbose=1, save_best_only=True, save_weights_only=False)
history = model.fit(ptrnsTrainX,ptrnsTrainY, batch_size=2, epochs=16,validation_data=(ptrnsValX,ptrnsValY),callbacks=[checkpointer],shuffle=True)
#history = model.fit(ptrnsTrainX,ptrnsTrainY,batch_size=8, epochs=16,validation_data=(ptrnsValX,ptrnsValY))#validation_split=0.1)#,shuffle=True)

model.save(modelName)


#Тестирование модели на конкретном изображении
#testModel(modelName,'train_data/93x93_2_2459.jpg',wsize)
#imgName='train_data/93x93_2_2459.jpg'
model = keras.models.load_model(modelName)
#model_b = load_model('tmp_model_/model_best_'+str(wsize)+'_'+str(out_size)+'_.h5')
imgName='test_data/94.jpg'
wsize=128
step=int(wsize)
if out_size==3:
     (imgBin1,imgBin2,imgBin3,prediction_,prediction)=testModel(model,imgName,wsize,step,pred_param,out_size,Ap_clahe)
else:
     (imgBin1,imgBin2,prediction_,prediction)=testModel_binary(model,imgName,wsize,step,pred_param,out_size,Ap_clahe)
a=prediction_[0,]
b=prediction[0,]

print(history.history.keys())

#История accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','validation'], loc='upper left')
plt.grid()
plt.show()
 
#loss история
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid()
plt.show()
#Выход:
# plt.plot(history.history["loss"], label="train_loss")
# plt.plot(history.history["val_loss"], label="val_loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Train and Validation Losses Over Epochs", fontsize=14)
# plt.legend()
# plt.grid()
# plt.show()


