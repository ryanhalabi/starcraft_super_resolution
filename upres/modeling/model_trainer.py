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
from tensorflow.keras.callbacks import Callback, TensorBoard
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
        self.validation_split = 0

    def train(self, images, epochs, batch_size, epochs_per_save, s3_sync=True):

        self.X = np.array([x.get_array(1 / self.sr_model.scaling) for x in images])

        self.Y = np.array([x.get_array() for x in images])
        if self.sr_model.rf_padding != 0:
            self.Y = self.Y[
                :,
                self.sr_model.rf_padding : -self.sr_model.rf_padding,
                self.sr_model.rf_padding : -self.sr_model.rf_padding,
                :,
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
        custom_callback = CustomCallback(
            self.X, self.sr_model, epochs_per_save, s3_sync
        )

        callbacks = [tensorboard_callback, custom_callback]

        self.sr_model.model.fit(
            self.X,
            self.Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=self.validation_split,
            verbose=1,
            callbacks=callbacks,
        )

    def log_initial_images(self):
        low_res = self.up_model.predict(self.X)
        # low_res = np.array([ cv2.resize(self.X[i,:,:,:], (self.Y.shape[2], self.Y.shape[1])) for i in range(self.X.shape[0])])

        log_images(self.Y, self.sr_model.name, self.sr_model.images_path, -2)
        log_images(low_res, self.sr_model.name, self.sr_model.images_path, -1)

    def set_upscale_model(self):
        """
        This sets up a baseline upscaler to compare visual results against.
        """
        inputs = keras.layers.Input(
            shape=(
                self.sr_model.input_shape[0],
                self.sr_model.input_shape[1],
                self.sr_model.channels,
            ),
            name="input",
        )

        bilin_interp = keras.layers.UpSampling2D(
            size=(self.sr_model.scaling, self.sr_model.scaling),
            interpolation="bilinear",
        )(inputs)

        rf_cropper = keras.layers.Cropping2D(cropping=self.sr_model.rf_padding)(
            bilin_interp
        )

        upscaler = keras.Model(inputs=inputs, outputs=rf_cropper)

        self.up_model = upscaler


class CustomCallback(tf.keras.callbacks.Callback):
    """
    Custom callback that saves the model and outputs images every 'epochs_per_save' epochs.
    """

    def __init__(self, X, sr_model, epochs_per_save, s3_sync):
        super().__init__()
        self.X = X
        self.model_path = sr_model.model_path
        self.images_path = sr_model.images_path
        self.model_name = sr_model.name

        self.start_epoch = sr_model.start_epoch
        self.epochs_per_save = epochs_per_save
        self.s3_sync = s3_sync

    def on_epoch_begin(self, epoch, logs=None):

        if epoch % self.epochs_per_save == 0:
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

            if self.s3_sync:
                env.sync_with_s3()


def log_images(images, model_name, images_path, epoch, start_epoch=0):
    file_writer = tf.summary.create_file_writer(
        str(images_path), filename_suffix=f"_{start_epoch + epoch}.v2"
    )

    with file_writer.as_default():
        tf.summary.image(
            model_name,
            images / 255,
            max_outputs=25,
            step=start_epoch + epoch,
        )

    for i in range(images.shape[0]):
        # open cv reads as BGR
        image = np.array(images[i, :, :, :])

        if image.shape[2] == 3:
            image[:, :, 0] = images[i, :, :, 2]
            image[:, :, 2] = images[i, :, :, 0]

        cv2.imwrite(f"{images_path}/static/{i}_{start_epoch + epoch}.jpg", image)
