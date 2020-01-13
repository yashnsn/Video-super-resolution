
  
from keras.layers import Activation, Dense, Input
# import pydot
from keras.layers import Conv2D, Flatten
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.optimizers import RMSprop
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model

from keras.layers.merge import concatenate

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import cv2
import os
import numpy as np
from keras.applications.vgg19 import VGG19
import keras.backend as K

def vgg_loss(y_true, y_pred):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=[1080, 1920, 3])
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def encoder_layer(inputs, filters=16, kernel_size=3, strides=1, activation='relu', instance_norm=True):
    
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x
  
def decoder_layer(inputs, paired_inputs, filters=16, kernel_size=3, strides=1, activation='relu', instance_norm=False):

    conv = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')
    x = inputs
    x1 = InstanceNormalization()(x)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = conv(x1)
    x2 = InstanceNormalization()(x1)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x2)
    x = concatenate([x1, x2, paired_inputs])
    return x
  
def build_generator(input_shape, output_shape, kernel_size=3, name=None):
    inputs_hr = Input(shape=output_shape)
    inputs_lr = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs_hr, 32, strides = 3, kernel_size=kernel_size, activation='leaky_relu')
    e1 = encoder_layer(e1, 64, activation='leaky_relu', kernel_size=kernel_size, strides = 1)
    e2 = encoder_layer(e1, 64, activation='leaky_relu', kernel_size=kernel_size, strides = 1)
    e2 = concatenate([e1,e2])
    e3 = encoder_layer(e2, 128, activation='leaky_relu',strides = 1, kernel_size=kernel_size)
    e3 = encoder_layer(e3, 128, activation='leaky_relu',strides = 1, kernel_size=kernel_size)
    e3 = concatenate([e2,e3])
    e4 = encoder_layer(e3, 3, activation='leaky_relu',strides = 1, kernel_size=kernel_size)
    e4 = encoder_layer(e4, 3, activation='leaky_relu',strides = 1, kernel_size=kernel_size)
    e4 = concatenate([e3,e4])
    e5 = encoder_layer(inputs_lr,3,activation='leaky_relu',strides = 2, kernel_size=kernel_size)
    inp = concatenate([e5, e4])
    d1 = decoder_layer(inp, e3, 128, kernel_size=kernel_size)
    d2 = decoder_layer(d1, e2, 64, kernel_size=kernel_size)
    d3 = decoder_layer(d2, e1, 32, kernel_size=kernel_size)
    d3 = Conv2DTranspose(channels, kernel_size=kernel_size, strides=3, padding='same')(d3)
    outputs = Conv2DTranspose(channels, kernel_size=kernel_size, strides=1, padding='same')(d3)
    generator = Model((inputs_hr,inputs_lr), outputs, name=name)

    return generator
  
def build_discriminator(input_shape, kernel_size=3, patchgan=True, name=None):

    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs, 16,strides=2, kernel_size=kernel_size, activation='leaky_relu', instance_norm=False)
    x = encoder_layer(x, 32,strides=2, kernel_size=kernel_size, activation='leaky_relu', instance_norm=False)
    x = encoder_layer(x, 64, kernel_size=kernel_size, activation='leaky_relu', strides=2, instance_norm=False)
    x = encoder_layer(x, 128, kernel_size=kernel_size, strides=2, activation='leaky_relu', instance_norm=False)

    if patchgan:
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(1, kernel_size=kernel_size, strides=2, padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)


    discriminator = Model(inputs, outputs, name=name)

    return discriminator
  
def build_gan():
    mov_shape = (720, 1280, 3)
    hr_shape = (1080, 1920 ,3)
    lr = 2e-4
    decay = 6e-8
    kernel_size = 3
    batch_size = 1
    gen = build_generator(mov_shape, hr_shape, kernel_size = kernel_size)
#     gen = load_model('gen_1080_vgg_1_1_0_1.h5', custom_objects={"InstanceNormalization": InstanceNormalization})
    gen.summary()
#     plot_model(gen, to_file='gen.png', show_shapes=True)
    disc = build_discriminator(hr_shape, kernel_size = kernel_size)
    disc.summary()
#     plot_model(disc, to_file='disc.png', show_shapes=True)
    optimizer = RMSprop(lr=0.01*lr, decay = 0.01*decay)
    disc.compile(loss = 'mse', optimizer = optimizer, metrics = ['accuracy'])
    disc.trainable = False
    mov_input = Input(shape = mov_shape)
    hr_input = Input(shape = hr_shape)
    hr_gen = gen([hr_input, mov_input])
    loss = ['mae',vgg_loss, 'mse']
    loss_weights = [0.4, 3, 10]
    inputs = [hr_input, mov_input]
    outputs = [gen([hr_input, mov_input]), gen([hr_input, mov_input]), disc(gen([hr_input, mov_input]))]
    adv = Model(inputs, outputs)
    optimizer = RMSprop(lr = lr, decay = decay)
    adv.compile(loss = loss, loss_weights = loss_weights, optimizer = optimizer, metrics = ['accuracy'])
    adv.summary()
#     plot_model(adv, to_file='adv.png', show_shapes=True)
    models = (gen, disc, adv)
    params = (batch_size, int(15000/batch_size), None)
    train_gan(models, params)
    
def train_gan(models, params):
    gen, disc, adv = models
    batch_size, train_steps, model_name = params
    save_interval = 1000
    patch = 34
    if patch > 1:
        d_patch = (34, 60, 1)
        valid = np.ones((batch_size,) + d_patch)
        fake = np.zeros((batch_size,) + d_patch)
    else:
        valid = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])

    valid_fake = np.concatenate((valid, fake))
#     start_time = datetime.now()
    count = 0
    cout=0
    for j in range(20):
        vidcap1 = cv2.VideoCapture('1080.mkv')
        vidcap3 = cv2.VideoCapture('1080.mkv')
        vidcap2 = cv2.VideoCapture('720.mkv')
        count = 0
        print(j)
        success = True
        for i in range(train_steps):
            vidcap2.read()
            vidcap3.read()
            imagesh1 = []
            imagesL1 = []
            imagesh2 = []
            count = 0
            while success and count<batch_size:
                print(1)
                success,image1 = vidcap1.read()
                imagesh1.append(image1)
                count += 1 
                success,image2 = vidcap2.read()
                imagesL1.append(image2)
                success,image3 = vidcap3.read()
                imagesh2.append(image3)
            hr_images = np.array(imagesh1)
            mov_images = np.array(imagesL1)
            hr1_images = np.array(imagesh2)
            gen_images = gen.predict([hr_images, mov_images]) #generated high resolution images
            gen_images = np.asarray(gen_images)
            x = np.concatenate((hr_images, gen_images)) #concatenating real and fake images to train discriminator
            y = np.ones([2*batch_size,1]) #outputs
            y[batch_size:, :]=0.0 # 1 for real and 0 for fake
            loss, acc = disc.train_on_batch(x,valid_fake) #training the discriminator
            log2 = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
            print(log2)
            y1 = np.ones([batch_size, 1])
            y=[hr1_images,hr1_images, valid] #output for adversial network
            x= [hr_images, mov_images] #input for adversial network
            log = adv.train_on_batch(x,y) #training adversial network
            print(log)
              
            if (i+1) % save_interval == 0:
                cout = cout+1
                if (i+1) == train_steps:
                    show = True
                else:
                    show = False
                gen.save(("gen_1080_inst_vgg_0.4_3_10_%d.h5")% (cout))
                disc.save(("disc_1080_inst_vgg_0.4_3_10_%d.h5")% (cout))
                
build_gan()
