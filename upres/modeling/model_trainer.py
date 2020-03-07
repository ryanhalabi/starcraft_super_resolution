# import matplotlib.pyplot as plt
import os
import re
import shutil
import uuid
import subprocess

import cv2
import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, Callback

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

    def train(self, images, epochs, epochs_per, log=True):
        train_images = images

        self.X = np.array(
            [x.get_array(1 / self.sr_model.scaling) for x in train_images]
        )

        self.Y = np.array([x.get_array() for x in train_images])
        if self.padding != 0:
            self.Y = self.Y[
                :, self.padding : -self.padding, self.padding : -self.padding, :
            ]

        if self.sr_model.start_epoch == 0:
            self.log_initial_images()

        epoch_log_path = self.sr_model.log_path / str(self.sr_model.start_epoch)
        if os.path.isdir(epoch_log_path):
            shutil.rmtree(epoch_log_path)
        os.mkdir(epoch_log_path)

        tensorboard_callback = TensorBoard(
            log_dir=str(epoch_log_path), histogram_freq=1
        )
        custom_callback = CustomCallback(self.X, self.sr_model, epochs_per)

        callbacks = [tensorboard_callback, custom_callback] if log else []

        self.sr_model.model.fit(
            self.X, self.Y, epochs=epochs, verbose=1, callbacks=callbacks,
        )

    def log_initial_images(self):
        low_res = self.up_model.predict(self.X)
        if self.padding != 0:
            low_res = low_res[
                :, self.padding : -self.padding, self.padding : -self.padding, :
            ]
        # low_res = np.array([ cv2.resize(self.X[i,:,:,:], (self.Y.shape[2], self.Y.shape[1])) for i in range(self.X.shape[0])])

        log_images(self.Y, self.sr_model.name, self.sr_model.images_path, -2)
        log_images(low_res, self.sr_model.name, self.sr_model.images_path, -1)

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


class CustomCallback(tf.keras.callbacks.Callback):
    """
    Custom callback that saves the model and outputs images every 'epochs_per' epochs.
    """

    def __init__(self, X, sr_model, epochs_per):
        super().__init__()
        self.X = X
        self.model_path = sr_model.model_path
        self.images_path = sr_model.images_path
        self.model_name = sr_model.name

        self.start_epoch = sr_model.start_epoch
        self.epochs_per = epochs_per

    def on_epoch_begin(self, epoch, logs=None):

        if epoch % self.epochs_per == 0:
            preds = self.model.predict(self.X)
            log_images(
                preds, self.model_name, self.images_path, epoch, self.start_epoch
            )

            self.model.save(
                str(
                    self.model_path
                    / "models"
                    / f"{self.model_name}_{self.start_epoch + epoch}.hdf5"
                )
            )


def log_images(images, model_name, images_path, epoch, start_epoch=0):
    file_writer = tf.summary.create_file_writer(
        str(images_path), filename_suffix=f"_{start_epoch + epoch}.v2"
    )

    with file_writer.as_default():
        tf.summary.image(
            model_name, images / 255, max_outputs=25, step=start_epoch + epoch,
        )

    env.sync_with_s3(env.data_path)
