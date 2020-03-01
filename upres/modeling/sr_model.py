import keras
import cv2
import numpy as np
import requests
import re
import pandas as pd

import os
from upres.utils.environment import env
import uuid
import shutil


class SRModel:
    """
    Customizable super resolution model
    """

    def __init__(self, name, input_shape, scaling=5, channels=1, conv_size=9, overwrite=False):

        self.name = name
        self.model_path = env.output / self.name
        self.scaling = scaling
        self.channels = channels
        self.conv_size = conv_size
        self.input_shape = input_shape
        self.set_optimizer()

        if overwrite and os.path.isdir(self.model_path):
            shutil.rmtree(self.model_path)

        self.create_or_load_model()

    def load_model(self, model_files):
        
        iteration = max([ int(re.findall(r"_(\d+).hdf5", x)[0]) for x in model_files])
        model_file_path = self.model_path / "models" / f"{self.name}_{iteration}.hdf5"

        self.iteration = iteration
        self.model = keras.models.load_model(str(model_file_path))


    def create_or_load_model(self):

        model_files = os.listdir(self.model_path / "models") if os.path.isdir(self.model_path / "models") else None
        if model_files:
            self.load_model(model_files)
        else:

            inputs = keras.layers.Input(shape=(self.input_shape[0], self.input_shape[1], self.channels))

            # upscaler
            upscaler = keras.layers.Conv2DTranspose(
                self.channels, (self.conv_size, self.conv_size), strides=(self.scaling, self.scaling),
            )(inputs)

            # trainable paramters
            conv_1 = keras.layers.Conv2D(64, (9, 9), strides=(1, 1), padding="same", activation="relu")(upscaler)
            conv_2 = keras.layers.Conv2D(32, (1, 1), strides=(1, 1), padding="same", activation="relu")(conv_1)
            conv_3 = keras.layers.Conv2D(self.channels, (5, 5), strides=(1, 1), padding="same")(conv_2)

            # combine upscale + trainable parameters, we predict residual
            add_layer = keras.layers.add([conv_3, upscaler])

            # add a depad layer, this removes the extra pixels on the edges
            depad_filter_size = 2 * self.conv_size - self.scaling
            depad_kernel = np.zeros([depad_filter_size, depad_filter_size, self.channels, self.channels])
            center = int(depad_kernel.shape[0] / 2)
            for i in range(self.channels):
                depad_kernel[center, center, i, i] = 1

            depad_bias = np.zeros([self.channels])

            predictions = keras.layers.Conv2D(
                self.channels,
                (depad_filter_size, depad_filter_size),
                strides=(1, 1),
                weights=[depad_kernel, depad_bias],
                trainable=False,
            )(add_layer)

            # finalize model
            model = keras.Model(inputs=inputs, outputs=predictions)
            model.compile(self.optimizer, "mean_squared_error")


            self.model = model
            self.iteration = 0

            if not os.path.isdir(self.model_path):
                os.mkdir(self.model_path)
                os.mkdir(self.model_path / "models")
                os.mkdir(self.model_path / "logs")
                os.mkdir(self.model_path / "images")
            self.save_model()


        self.log_path = self.model_path / "logs"
        self.images_path = self.model_path / "images"

        print(f"Loaded model {self.name}, iteration {self.iteration}.")

    def set_optimizer(self):
        self.optimizer = keras.optimizers.Adam()

    def save_model(self):
        self.model.save(str(self.model_path / "models" / f"{self.name}_{self.iteration}.hdf5"))
