from keras.models import Model
from keras.layers import *


WEIGHT_INITIALIZER = "he_normal"
CONV_PADDING = "SAME"
CONV_KERNEL_SIZE = (3,3)
CONV_ACTIVATION = "relu"
POOLING_SIZE = (2,2)
DECONV_STRIDES = (2,2)
DROPOUT_RATE = 0.5


def convolutional_block(input_tensor, units):
    x = Conv2D(filters=units, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING, kernel_initializer=WEIGHT_INITIALIZER)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(CONV_ACTIVATION)(x)
    x = Conv2D(filters=units, kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING, kernel_initializer=WEIGHT_INITIALIZER)(x)
    x = BatchNormalization()(x)
    x = Activation(CONV_ACTIVATION)(x)
    return x


def deconvolutional_block(input_tensor, residual, units):
    x = Conv2DTranspose(units, kernel_size=CONV_KERNEL_SIZE, strides=DECONV_STRIDES, padding=CONV_PADDING)(input_tensor)
    # add parallel residual from other side of network
    x = concatenate([x, residual], axis=3)
    x = convolutional_block(x, units)
    return x


def Unet(img_height, img_width, nclasses=3, filters=64):
    # input
    input_layer = Input(shape=(img_height, img_width, 3))

    # down 1
    conv1 = convolutional_block(input_layer, filters)
    conv1_out = MaxPooling2D(pool_size=POOLING_SIZE)(conv1)

    # down 2
    conv2 = convolutional_block(conv1_out, filters*2)
    conv2_out = MaxPooling2D(pool_size=POOLING_SIZE)(conv2)

    # down 3
    conv3 = convolutional_block(conv2_out, filters*4)
    conv3_out = MaxPooling2D(pool_size=POOLING_SIZE)(conv3)

    # down 4
    conv4 = convolutional_block(conv3_out, filters*8)
    conv4_out = MaxPooling2D(pool_size=POOLING_SIZE)(conv4)
    conv4_out = Dropout(DROPOUT_RATE)(conv4_out)

    # down 5
    conv5 = convolutional_block(conv4_out, filters*16)
    conv5 = Dropout(DROPOUT_RATE)(conv5)

    # up 1
    deconv6 = deconvolutional_block(conv5, conv4, filters*8)
    deconv6 = Dropout(DROPOUT_RATE)(deconv6)

    # up 2
    deconv7 = deconvolutional_block(deconv6, conv3, filters*4)
    deconv7 = Dropout(DROPOUT_RATE)(deconv7)

    # up 3
    deconv8 = deconvolutional_block(deconv7, conv2, filters*2)

    # up 4
    deconv9 = deconvolutional_block(deconv8, conv1, filters)

    # output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
    output_layer = BatchNormalization()(output_layer)
    output_layer = Activation('softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model

m = Unet(128,128)