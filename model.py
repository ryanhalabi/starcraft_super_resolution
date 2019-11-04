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





class Model:

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
        self.set_loss()

        if os.path.isfile(env.path / 'models' / f"{name}.hdf5"):
            self.model = keras.models.load_model( str(env.path / 'models' / f"{name}.hdf5"))
        else:


            inputs = keras.layers.Input(shape=(100, 100, self.channels))
            upscaler = keras.layers.Conv2DTranspose(self.channels, (self.conv_size, self.conv_size), strides=(self.scaling,self.scaling))(inputs)

            conv_1 = keras.layers.Conv2D(32, (9,9), strides=(1,1), padding='same', activation='relu')(upscaler)
            conv_2 = keras.layers.Conv2D(16, (1,1), strides=(1,1), padding='same', activation='relu')(conv_1)
            conv_3 = keras.layers.Conv2D(self.channels, (5,5), strides=(1,1), padding='same')(conv_2)
            
            add_layer = keras.layers.add([conv_3, upscaler])

            depad_filter_size = 2*self.conv_size - self.scaling
            depad_kernel = np.zeros([depad_filter_size, depad_filter_size,1,1])
            center = int(depad_kernel.shape[0]/2)
            depad_kernel[center, center,0,0] = 1
            depad_bias = np.zeros([1])

            predictions = keras.layers.Conv2D(self.channels, (depad_filter_size, depad_filter_size), strides=(1,1),
                                            weights=[depad_kernel, depad_bias], trainable=False)(add_layer)

            model = keras.Model(inputs=inputs, outputs=predictions)

            model.compile(self.optimizer, self.loss)

            self.model = model

        if os.path.isdir(env.path / 'data' / self.name):
            shutil.rmtree(env.path / 'data' / self.name)
        os.mkdir( env.path / 'data' / self.name )

    def set_optimizer(self):
        self.optimizer = keras.optimizers.Adam()

    def set_loss(self):

        self.loss='mean_squared_error'
        
        # def custom_loss(y_pred, y_true):
        #     # zeros = keras.backend.zeros(y_pred.shape)
        #     # import pdb
        #     # pdb.set_trace()
            
        #     return keras.backend.abs( (y_true - y_pred) )
        #     # keras.backend.

        # self.loss = custom_loss


    def save_model(self):
        self.model.save( str(env.path / 'models' / f"{self.model.name}.hdf5"))


class TrainSRModel:

    def __init__(self, model, save=False):
        self.model = model
        self.save = save

        self.set_up_model()

    def set_up_model(self):
        up_model = keras.Sequential()
        up_model.add(keras.layers.UpSampling2D(size=(self.model.scaling,self.model.scaling), interpolation='bilinear', input_shape=(None, None, self.model.channels)))
        self.up_model = up_model


    def train(self, images, epochs, save):
        train_images = images[0:1]

        self.X = np.array([x.get_array(self.model.scaling) for x in train_images])

        y = [x.get_array() for x in train_images]
        padding = int((self.model.conv_size-1)/2)
        y = [x[padding:-padding, padding:-padding,:] for x in y]
        self.Y = np.array(y)

        self.save_initial(images)

        for i in range(epochs):
            self.predict(images, i)
            self.model.model.fit(self.X, self.Y, epochs = save) 
        self.predict(images, '0_final')

        if self.save:
            self.model.save(f"{self.model.name}.hdf5")

    def predict(self, images, name):

        x = np.array([x.get_array(self.model.scaling) for x in images])
        image_names = [x.name for x in images]

        preds = self.model.model.predict(x)

        for j, pred in enumerate(preds):
            cv2.imwrite( str(env.path / 'data' / self.model.name / f"{image_names[j]}_{name}.png"), pred)



    def save_initial(self, images):

        image_names = [x.name for x in images]

        y = [x.get_array() for x in images]
        padding = int((self.model.conv_size-1)/2)
        y = [x[padding:-padding, padding:-padding,:] for x in y]
        Y = np.array(y)

        for j, y in enumerate(Y):
            cv2.imwrite( str(env.path / 'data' / self.model.name / f"{image_names[j]}_0_target.png"), y)

        X = np.array([x.get_array(self.model.scaling) for x in images])

        up_samples = self.up_model.predict(X)
        padding = int((self.model.conv_size-1)/2)
        up_samples = [x[padding:-padding, padding:-padding,:] for x in up_samples]

        for j, up_sample in enumerate(up_samples):
            # up_sample = np.pad(up_sample, paddings, mode='constant')
            cv2.imwrite( str(env.path / 'data' / self.model.name / f"{image_names[j]}_0_upscaled.png"), up_sample)




    # def plot_filters(self):

    #     weights = self.model.weights
    #     for weight in weights:
    #         for i in range( weight.shape[2]):
    #             filter = weight[:,:,i,0]
    #             plt.plot(filter)
    #             plt.show()





