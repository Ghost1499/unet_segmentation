import keras
from keras import backend as K
from keras.layers import (
    Activation,
    Add,
    Conv2D,
    Input,
    Layer,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    BatchNormalization,
    SpatialDropout2D,
    Resizing,
)
from keras.models import Model  # ,MaxPooling2D, UpSampling2D, concatenate
from keras.regularizers import l2
import tensorflow as tf

from configs import io_config


def double_conv_layer(x, Nc, filters, l2_val, dropout_val, batch_norm):
    if K.image_data_format() == "channels_last":  # определение используемого бэкенда
        axis = 3
    else:
        axis = 1

    conv = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(l2_val))(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation("relu")(conv)
    if Nc >= 2:
        conv = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(l2_val))(
            conv
        )
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation("relu")(conv)
    if Nc >= 3:
        conv = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(l2_val))(
            conv
        )
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation("relu")(conv)

    if Nc >= 4:
        conv = Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(l2_val))(
            conv
        )
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation("relu")(conv)
    if dropout_val > 0.0:
        conv = SpatialDropout2D(dropout_val)(conv)
    return conv


class my_layer(Layer):
    def __init__(self, por, **kwargs):  # инициализация переменной класса и суперкласса
        self.por = por
        super(my_layer, self).__init__(**kwargs)

    def build(self, input_shape):  # опредаление настраиваемых перменных
        self.w1 = self.add_weight(
            name="w1",
            shape=(input_shape[3],),
            #         initializer=keras.initializers.Constant(value=0.10),trainable=True)
            initializer=keras.initializers.RandomUniform(minval=0.25, maxval=1.00),
            trainable=True,
        )
        # self.w2 = self.add_weight(name='w2',shape=(input_shape[3],),
        #           initializer=keras.initializers.RandomUniform(minval=0.5,maxval=1.00),trainable=True)
        # self.por = self.add_weight(name='por',shape=(input_shape[3],),
        #            #initializer=keras.initializers.Constant(value=64.0),trainable=True)
        #            initializer=keras.initializers.RandomUniform(minval=0.0,maxval=0.5),trainable=True)
        super(my_layer, self).build(input_shape)
        # self.built = True

    def call(self, x, y):  # проводимые вычисления
        y_ = tf.where(y < self.por, 0.0, y)
        # y__=tf.where(y_>2*self.por,y_,1)
        # out_put=x*self.w1+y_
        out_put = x * self.w1 + y_
        # print(self.w)
        return out_put

    def compute_output_shape(self):  # определяется форма выхода
        return self.out_put

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "por": self.por,  # сериализация сохраняет в том числе параметры слоя
            }
        )


def createUNetModel_My_AB(
    model_name, ptrnShape, Nc, filters, out_size, l2_val, dropout_val, por, batch_norm
):
    # Здесь используется функциональная модель API для нелинейных взаимодействия межуд слоями
    # Разница заключается в том, что входной слой для последовательной модели создается и применяется неявно
    # (и поэтому не будет доступен через атрибут .layers ),
    # тогда как для моделей, построенных с помощью Functional API,
    # необходимо явно определить входной слой

    if K.image_data_format() == "channels_last":
        inputs_a = Input(ptrnShape)
        inputs_b = Input(ptrnShape)
        axis = 3
    else:
        ptrnShape = tuple(reversed(ptrnShape))  # перевертывание кортежа
        inputs_a = Input(ptrnShape)
        inputs_b = Input(ptrnShape)
        axis = 1

    # outputs_b_1=unet_bin_layer(inputs_b,Nc,filters,l2_val,dropout_val,batch_norm)

    # inputs_a_=Conv2D(1, (5, 5),padding='same',activation='linear')(inputs_a)
    #                 kernel_initializer=keras.initializers.Constant(value=0.75),bias_initializer='zeros',
    #                 kernel_regularizer=l2(l2_val))(inputs_a)
    # inputs_a_=LocallyConnected2D(1,(1, 1),padding='same',activation='linear',kernel_regularizer=l2(l2_val))(inputs_a)
    # inputs_b_=Conv2D(1, (5, 5),padding='same',activation='linear')(inputs_b)
    #                 kernel_initializer=keras.initializers.Constant(value=1.00),bias_initializer='zeros',
    x = Add()([inputs_a, inputs_b])

    # x = concatenate([inputs_a, inputs_b],axis=axis)
    # x=my_layer(por)(inputs_a,inputs_b)
    conv_1 = double_conv_layer(x, Nc, filters, l2_val, dropout_val, batch_norm)
    down_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1)
    conv_2 = double_conv_layer(down_1, Nc, 2 * filters, l2_val, dropout_val, batch_norm)
    down_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = double_conv_layer(down_2, Nc, 4 * filters, l2_val, dropout_val, batch_norm)
    down_3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_3)

    conv_4 = double_conv_layer(down_3, Nc, 8 * filters, l2_val, dropout_val, batch_norm)
    down_4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_4)

    conv_5 = double_conv_layer(
        down_4, Nc, 16 * filters, l2_val, dropout_val, batch_norm
    )
    down_5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_5)

    conv_6 = double_conv_layer(
        down_5, Nc, 32 * filters, l2_val, dropout_val, batch_norm
    )

    up_1 = UpSampling2D((2, 2))(conv_6)
    conv_71 = Conv2D(
        16 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_1)
    concat_1 = concatenate([conv_5, conv_71], axis=axis)
    conv_7 = double_conv_layer(
        concat_1, Nc, 16 * filters, l2_val, dropout_val, batch_norm
    )

    up_2 = UpSampling2D((2, 2))(conv_7)
    conv_81 = Conv2D(
        8 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_2)
    concat_2 = concatenate([conv_4, conv_81], axis=axis)
    conv_8 = double_conv_layer(
        concat_2, Nc, 8 * filters, l2_val, dropout_val, batch_norm
    )

    up_3 = UpSampling2D((2, 2))(conv_8)
    conv_91 = Conv2D(
        4 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_3)
    concat_3 = concatenate([conv_3, conv_91], axis=axis)
    conv_9 = double_conv_layer(
        concat_3, Nc, 4 * filters, l2_val, dropout_val, batch_norm
    )

    up_4 = UpSampling2D((2, 2))(conv_9)
    conv_101 = Conv2D(
        2 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_4)
    concat_4 = concatenate([conv_2, conv_101], axis=axis)
    conv_10 = double_conv_layer(
        concat_4, Nc, 2 * filters, l2_val, dropout_val, batch_norm
    )

    up_5 = UpSampling2D((2, 2))(conv_10)
    conv_111 = Conv2D(
        filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_5)
    concat_5 = concatenate([conv_1, conv_111], axis=axis)
    conv_11 = double_conv_layer(concat_5, Nc, filters, l2_val, dropout_val, batch_norm)

    if out_size == 1:
        conv_12a = Conv2D(
            out_size, (1, 1), padding="same", kernel_regularizer=l2(l2_val)
        )(conv_11)
        outputs_a = Activation("sigmoid")(conv_12a)
        conv_12b = Conv2D(
            out_size, (1, 1), padding="same", kernel_regularizer=l2(l2_val)
        )(conv_11)
        outputs_b_2 = Activation("sigmoid")(conv_12b)
    else:
        conv_12a = Conv2D(
            out_size, (1, 1), padding="same", kernel_regularizer=l2(l2_val)
        )(conv_11)
        outputs_a = Activation("softmax")(conv_12a)
        conv_12b = Conv2D(
            out_size, (1, 1), padding="same", kernel_regularizer=l2(l2_val)
        )(conv_11)
        outputs_b_2 = Activation("softmax")(conv_12b)
    # outputs_b = concatenate([outputs_b_1,outputs_b_2], axis=axis)
    # model = Model(inputs_b,conv_10)
    model = Model([inputs_a, inputs_b], [outputs_a, outputs_b_2])

    return model


def createUNetModel_My_A(
    model_name, ptrnShape, Nc, filters, out_size, l2_val, dropout_val, batch_norm
):
    # Здесь используется функциональная модель API для нелинейных взаимодействия межуд слоями
    # Разница заключается в том, что входной слой для последовательной модели создается и применяется неявно
    # (и поэтому не будет доступен через атрибут .layers ),
    # тогда как для моделей, построенных с помощью Functional API,
    # необходимо явно определить входной слой

    if K.image_data_format() == "channels_last":
        inputs_a = Input(ptrnShape)
        axis = 3
    else:
        ptrnShape = tuple(reversed(ptrnShape))  # перевертывание кортежа
        inputs_a = Input(ptrnShape)
        axis = 1
    conv_1 = double_conv_layer(inputs_a, Nc, filters, l2_val, dropout_val, batch_norm)
    down_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1)
    conv_2 = double_conv_layer(down_1, Nc, 2 * filters, l2_val, dropout_val, batch_norm)
    down_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = double_conv_layer(down_2, Nc, 4 * filters, l2_val, dropout_val, batch_norm)
    down_3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_3)

    conv_4 = double_conv_layer(down_3, Nc, 8 * filters, l2_val, dropout_val, batch_norm)
    down_4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_4)

    conv_5 = double_conv_layer(
        down_4, Nc, 16 * filters, l2_val, dropout_val, batch_norm
    )
    down_5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_5)

    conv_6 = double_conv_layer(
        down_5, Nc, 32 * filters, l2_val, dropout_val, batch_norm
    )

    up_1 = UpSampling2D((2, 2))(conv_6)
    conv_71 = Conv2D(
        16 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_1)
    concat_1 = concatenate([conv_5, conv_71], axis=axis)
    conv_7 = double_conv_layer(
        concat_1, Nc, 16 * filters, l2_val, dropout_val, batch_norm
    )

    up_2 = UpSampling2D((2, 2))(conv_7)
    conv_81 = Conv2D(
        8 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_2)
    concat_2 = concatenate([conv_4, conv_81], axis=axis)
    conv_8 = double_conv_layer(
        concat_2, Nc, 8 * filters, l2_val, dropout_val, batch_norm
    )

    up_3 = UpSampling2D((2, 2))(conv_8)
    conv_91 = Conv2D(
        4 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_3)
    concat_3 = concatenate([conv_3, conv_91], axis=axis)
    conv_9 = double_conv_layer(
        concat_3, Nc, 4 * filters, l2_val, dropout_val, batch_norm
    )

    up_4 = UpSampling2D((2, 2))(conv_9)
    conv_101 = Conv2D(
        2 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_4)
    concat_4 = concatenate([conv_2, conv_101], axis=axis)
    conv_10 = double_conv_layer(
        concat_4, Nc, 2 * filters, l2_val, dropout_val, batch_norm
    )

    up_5 = UpSampling2D((2, 2))(conv_10)
    conv_111 = Conv2D(
        filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_5)
    concat_5 = concatenate([conv_1, conv_111], axis=axis)
    conv_11 = double_conv_layer(concat_5, Nc, filters, l2_val, dropout_val, batch_norm)

    if out_size == 1:
        conv_12 = Conv2D(1, (1, 1), padding="same", kernel_regularizer=l2(l2_val))(
            conv_11
        )
        outputs_a = Activation("sigmoid")(conv_12)
    else:
        conv_12b = Conv2D(
            out_size, (1, 1), padding="same", kernel_regularizer=l2(l2_val)
        )(conv_11)
        outputs_a = Activation("softmax")(conv_12b)

    # model = Model(inputs_b,conv_10)
    model = Model(inputs_a, outputs_a)

    return model


def createUNetModel_My_All(
    ptrnShape, Nc, filters, out_size, l2_val, dropout_val, batch_norm
):
    # Здесь используется функциональная модель API для нелинейных взаимодействия межуд слоями
    # Разница заключается в том, что входной слой для последовательной модели создается и применяется неявно
    # (и поэтому не будет доступен через атрибут .layers ),
    # тогда как для моделей, построенных с помощью Functional API,
    # необходимо явно определить входной слой

    if K.image_data_format() == "channels_last":
        inputs_a = Input(ptrnShape)
        inputs_b = Input(ptrnShape)
        axis = 3
    else:
        ptrnShape = tuple(reversed(ptrnShape))  # перевертывание кортежа
        inputs_a = Input(ptrnShape)
        inputs_b = Input(ptrnShape)
        axis = 1

    conv_1s = double_conv_layer(inputs_a, Nc, filters, l2_val, dropout_val, batch_norm)
    down_1s = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1s)

    conv_1e = double_conv_layer(inputs_b, Nc, filters, l2_val, dropout_val, batch_norm)
    down_1e = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1e)

    down_1 = my_layer(por=0.0)(down_1s, down_1e)

    conv_2s = double_conv_layer(
        down_1s, Nc, 2 * filters, l2_val, dropout_val, batch_norm
    )
    down_2s = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_2s)

    conv_2e = double_conv_layer(
        down_1, Nc, 2 * filters, l2_val, dropout_val, batch_norm
    )
    down_2e = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_2e)

    down_2 = my_layer(por=0.0)(down_2s, down_2e)

    conv_3s = double_conv_layer(
        down_2s, Nc, 4 * filters, l2_val, dropout_val, batch_norm
    )
    down_3s = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_3s)

    conv_3e = double_conv_layer(
        down_2, Nc, 4 * filters, l2_val, dropout_val, batch_norm
    )
    down_3e = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_3e)

    down_3 = my_layer(por=0.0)(down_3s, down_3e)

    conv_4s = double_conv_layer(
        down_3s, Nc, 8 * filters, l2_val, dropout_val, batch_norm
    )
    down_4s = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_4s)

    conv_4e = double_conv_layer(
        down_3, Nc, 8 * filters, l2_val, dropout_val, batch_norm
    )
    down_4e = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_4e)

    down_4 = my_layer(por=0.0)(down_4s, down_4e)

    conv_5s = double_conv_layer(
        down_4s, Nc, 16 * filters, l2_val, dropout_val, batch_norm
    )
    down_5s = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_5s)

    conv_5e = double_conv_layer(
        down_4, Nc, 16 * filters, l2_val, dropout_val, batch_norm
    )
    down_5e = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_5e)

    # down_5=my_layer(por=0.0)(down_5s,down_5e)

    # conv_6s = double_conv_layer(down_5s,Nc,32*filters,l2_val,dropout_val,batch_norm)
    # conv_6e = double_conv_layer(down_5,Nc,32*filters,l2_val,dropout_val,batch_norm)
    # concat_6es = concatenate([conv_6s, conv_6e], axis=axis)

    concat_5es = concatenate([down_5s, down_5e], axis=axis)

    up_1 = UpSampling2D((2, 2))(concat_5es)
    conv_71 = Conv2D(
        16 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_1)
    conv_7 = double_conv_layer(
        conv_71, Nc, 16 * filters, l2_val, dropout_val, batch_norm
    )
    concat_1 = concatenate([down_4, conv_7], axis=axis)

    up_2 = UpSampling2D((2, 2))(concat_1)
    conv_81 = Conv2D(
        8 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_2)
    conv_8 = double_conv_layer(
        conv_81, Nc, 8 * filters, l2_val, dropout_val, batch_norm
    )
    concat_2 = concatenate([down_3, conv_8], axis=axis)

    up_3 = UpSampling2D((2, 2))(concat_2)
    conv_91 = Conv2D(
        4 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_3)
    conv_9 = double_conv_layer(
        conv_91, Nc, 4 * filters, l2_val, dropout_val, batch_norm
    )
    concat_3 = concatenate([down_2, conv_9], axis=axis)

    up_4 = UpSampling2D((2, 2))(concat_3)
    conv_101 = Conv2D(
        2 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_4)
    conv_10 = double_conv_layer(
        conv_101, Nc, 2 * filters, l2_val, dropout_val, batch_norm
    )
    concat_2 = concatenate([down_1, conv_10], axis=axis)

    up_5 = UpSampling2D((2, 2))(concat_2)
    conv_111 = Conv2D(
        filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_5)
    conv_11 = double_conv_layer(conv_111, Nc, filters, l2_val, dropout_val, batch_norm)
    # concat_5 = concatenate([down_1, conv_11], axis=axis)

    if out_size == 1:
        conv_12 = Conv2D(1, (1, 1), padding="same", kernel_regularizer=l2(l2_val))(
            conv_11
        )
        outputs_a = Activation("sigmoid")(conv_12)
    else:
        conv_12b = Conv2D(
            out_size, (1, 1), padding="same", kernel_regularizer=l2(l2_val)
        )(conv_11)
        outputs_a = Activation("softmax")(conv_12b)

    # model = Model(inputs_b,conv_10)
    model = Model([inputs_a, inputs_b], outputs_a)

    return model


def createUNetModel_My(
    input_shape, Nc, filters, out_size, l2_val, dropout_val, batch_norm
):
    # Здесь используется функциональная модель API для нелинейных взаимодействия межуд слоями
    # Разница заключается в том, что входной слой для последовательной модели создается и применяется неявно
    # (и поэтому не будет доступен через атрибут .layers ),
    # тогда как для моделей, построенных с помощью Functional API,
    # необходимо явно определить входной слой

    if K.image_data_format() == "channels_last":
        axis = 3
    else:
        input_shape = tuple(reversed(input_shape))  # перевертывание кортежа
        axis = 1
    inputs = Input(input_shape)

    # resize_1 = Resizing(target_shape[0],target_shape[1])(inputs)
    conv_1 = double_conv_layer(inputs, Nc, filters, l2_val, dropout_val, batch_norm)
    down_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1)
    conv_2 = double_conv_layer(down_1, Nc, 2 * filters, l2_val, dropout_val, batch_norm)
    down_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_2)

    conv_3 = double_conv_layer(down_2, Nc, 4 * filters, l2_val, dropout_val, batch_norm)
    down_3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_3)

    conv_4 = double_conv_layer(down_3, Nc, 8 * filters, l2_val, dropout_val, batch_norm)
    down_4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_4)

    conv_5 = double_conv_layer(
        down_4, Nc, 16 * filters, l2_val, dropout_val, batch_norm
    )

    up_1 = UpSampling2D((2, 2))(conv_5)
    conv_61 = Conv2D(
        8 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_1)
    concat_1 = concatenate([conv_4, conv_61], axis=axis)
    conv_6 = double_conv_layer(
        concat_1, Nc, 8 * filters, l2_val, dropout_val, batch_norm
    )

    up_2 = UpSampling2D((2, 2))(conv_6)
    conv_71 = Conv2D(
        4 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_2)
    concat_2 = concatenate([conv_3, conv_71], axis=axis)
    conv_7 = double_conv_layer(
        concat_2, Nc, 4 * filters, l2_val, dropout_val, batch_norm
    )

    up_3 = UpSampling2D((2, 2))(conv_7)
    conv_81 = Conv2D(
        2 * filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_3)
    concat_3 = concatenate([conv_2, conv_81], axis=axis)
    conv_8 = double_conv_layer(
        concat_3, Nc, 2 * filters, l2_val, dropout_val, batch_norm
    )

    up_4 = UpSampling2D((2, 2))(conv_8)
    conv_91 = Conv2D(
        filters,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(l2_val),
    )(up_4)
    concat_4 = concatenate([conv_1, conv_91], axis=axis)
    conv_9 = double_conv_layer(concat_4, Nc, filters, l2_val, dropout_val, batch_norm)

    if out_size == 1:
        conv_10 = Conv2D(1, (1, 1), padding="same", kernel_regularizer=l2(l2_val))(
            conv_9
        )
        conv_10 = Activation("sigmoid")(conv_10)
    else:
        conv_10 = Conv2D(
            out_size, (1, 1), padding="same", kernel_regularizer=l2(l2_val)
        )(conv_9)
        conv_10 = Activation("softmax")(conv_10)

    return Model(inputs, conv_10)


def load_model_arch(model_name):
    with open(io_config.MODEL_SAVE_DIR / f"{model_name}_architecture.json") as f:
        return f.read()


def load_model(model_name):
    return keras.saving.load_model(
        io_config.MODEL_SAVE_DIR / f"{model_name}.keras",
        custom_objects=None,
        compile=True,
        safe_mode=True,
    )
