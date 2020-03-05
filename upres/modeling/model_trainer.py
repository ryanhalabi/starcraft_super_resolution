# import matplotlib.pyplot as plt
import os
import re
import shutil
import uuid

import cv2
import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

from upres.modeling.sr_model import SRModel
from upres.utils.environment import env
from upres.utils.image import Image


class ModelTrainer:
    """
    Handles training models and storing results.
    """

    def __init__(self, sr_model):
        self.sr_model = sr_model
        self.set_upscale_model()

        # The # of points we truncate off the edges of the NN output.
        self.padding = int((self.sr_model.max_kernel_size - 1) / 2)

    def train(self, images, epochs, batches, log=True):
        train_images = images

        self.X = np.array(
            [x.get_array(1 / self.sr_model.scaling) for x in train_images]
        )

        y = np.array([x.get_array() for x in train_images])
        self.Y = y[:, self.padding : -self.padding, self.padding : -self.padding, :]

        if self.sr_model.iteration == 0:
            self.log_initial_images()

        for _ in range(batches):
            iteration_path = self.sr_model.log_path / str(self.sr_model.iteration)
            if os.path.isdir(iteration_path):
                shutil.rmtree(iteration_path)
            os.mkdir(iteration_path)

            tensorboard_callback = TensorBoard(
                log_dir=str(iteration_path), histogram_freq=1
            )
            callbacks = [tensorboard_callback] if log else []

            self.sr_model.model.fit(
                self.X, self.Y, epochs=epochs, verbose=1, callbacks=callbacks,
            )

            if log:
                self.predict(images, self.sr_model.iteration)
            print(
                f"Model iteration {self.sr_model.iteration} Loss: {self.sr_model.model.history.history['loss'][-1]}\n\n"
            )

            self.sr_model.iteration += 1
            self.sr_model.save_model()

    def predict(self, images, step):
        x = np.array([x.get_array(1 / self.sr_model.scaling) for x in images])
        preds = self.sr_model.model.predict(x)

        self.log_images(preds)

        return preds

    def log_initial_images(self):
        low_res = self.up_model.predict(self.X)
        low_res = low_res[
            :, self.padding : -self.padding, self.padding : -self.padding, :
        ]
        # low_res = np.array([ cv2.resize(self.X[i,:,:,:], (self.Y.shape[2], self.Y.shape[1])) for i in range(self.X.shape[0])])
        self.log_images(low_res, override_step=-2)
        self.log_images(self.Y, override_step=-1)

    def log_images(self, images, override_step=None):
        step = override_step if override_step else self.sr_model.iteration
        file_writer = tf.summary.create_file_writer(str(self.sr_model.images_path))

        # # for some reason tensorboard is BGR not RGB?
        # x = np.copy(images)
        # images[:, :, :, 0] = x[:, :, :, 2]
        # images[:, :, :, 2] = x[:, :, :, 0]

        with file_writer.as_default():
            tf.summary.image(
                self.sr_model.name, images / 255, max_outputs=25, step=step
            )

    def set_upscale_model(self):
        """
        This sets up a simple upscaling CNN to compare against.
        """
        up_model = keras.Sequential()
        up_model.add(
            keras.layers.UpSampling2D(
                size=(self.sr_model.scaling, self.sr_model.scaling),
                interpolation="bilinear",
                input_shape=(None, None, self.sr_model.channels),
            )
        )

        self.up_model = up_model

    # def plot_filters(self):

    #     weights = self.model.weights
    #     for weight in weights:
    #         for i in range( weight.shape[2]):
    #             filter = weight[:,:,i,0]
    #             plt.plot(filter)
    #             plt.show()

    # def save_initial(self, images):
    #     image_names = [x.name for x in images]

    #     y = [x.get_array() for x in images]
    #     padding = (self.model.conv_size - 1) // 2
    #     y = [x[padding:-padding, padding:-padding, :] for x in y]
    #     Y = np.array(y)

    #     X = np.array([x.get_array(1 / self.model.scaling) for x in images])

    #     up_samples = self.up_model.predict(X)
    #     padding = int((self.model.conv_size - 1) / 2)
    #     up_samples = np.array(
    #         [x[padding:-padding, padding:-padding, :] for x in up_samples]
    #     )

    #     self.log_images(Y, 0)
    #     self.log_images(up_samples, 0)
