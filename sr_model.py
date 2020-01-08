import keras
import cv2
import numpy as np
import requests
import re
# import matplotlib.pyplot as plt
import os
from environment import env
import uuid
import shutil





class SRModel:

    def __init__(self, name, scaling=5, channels=1, conv_size=9):
        self.name = name
        self.scaling = scaling
        self.channels = channels
        self.conv_size = conv_size

        # scaling x the pixels, (conv_size - scaling)/2 extra pixels get added on each side as padding from conv2dtranspose
        # we dont care about these pixels for comparing against target
        # we also dont care about the pixels influenced by these extra pixels
        # get rid of depad filter of size 
        # (conv_size - scaling + 1) - gets rid of pad pixels
        # (conv_size+1) - gets rid of pixels influenced by padded pixels
        # so get rid of 2*conv_size - scaling pixels total

        # ex, 5x, conv 9
        # 0 0 x x X x+y x+y Y y+w y+w W
        # 18 - 5 = 13 = gets rid of 6 pixels each side
        # im probably fucking this up since further conv layers increase the impact of pixels
        # oh well

        self.set_optimizer()

        if not os.path.isdir(env.output_data / 'models' / self.name):
            os.mkdir( env.output_data / 'models' / self.name )
        for file in os.listdir( env.output_data / 'models' / self.name):
          os.remove(  env.output_data / 'models' / self.name / file)
        

        if os.path.isfile(env.output_data / 'models' / f"{name}.hdf5"):
            self.model = keras.models.load_model( str(env.path / 'models' / f"{name}.hdf5"))
        else:


            inputs = keras.layers.Input(shape=(100, 100, self.channels))
            upscaler = keras.layers.Conv2DTranspose(self.channels, (self.conv_size, self.conv_size), strides=(self.scaling,self.scaling))(inputs)

            conv_1 = keras.layers.Conv2D(64, (9,9), strides=(1,1), padding='same', activation='relu')(upscaler)
            conv_2 = keras.layers.Conv2D(32, (1,1), strides=(1,1), padding='same', activation='relu')(conv_1)
            conv_3 = keras.layers.Conv2D(self.channels, (5,5), strides=(1,1), padding='same')(conv_2)
            
            add_layer = keras.layers.add([conv_3, upscaler])

            depad_filter_size = 2*self.conv_size - self.scaling
            depad_kernel = np.zeros([depad_filter_size, depad_filter_size, self.channels, self.channels])
            center = int(depad_kernel.shape[0]/2)
            for i in range(self.channels):
              depad_kernel[center, center,i,i] = 1

            depad_bias = np.zeros([self.channels])

            predictions = keras.layers.Conv2D(self.channels, (depad_filter_size, depad_filter_size), strides=(1,1),
                                            weights=[depad_kernel, depad_bias], trainable=False)(add_layer)
            # predictions = keras.layers.Conv2D(self.channels, (depad_filter_size, depad_filter_size), strides=(1,1))(add_layer)

            model = keras.Model(inputs=inputs, outputs=predictions)

            model.compile(self.optimizer, 'mean_squared_error')

            self.model = model


    def set_optimizer(self):
        self.optimizer = keras.optimizers.Adam()


    def save_model(self):
        self.model.save( str(env.path / 'models' / f"{self.model.name}.hdf5"))

