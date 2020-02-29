import keras
import cv2
import numpy as np
import requests
import re

# import matplotlib.pyplot as plt
import os
from upres.utils.environment import env
from upres.modeling.sr_model import SRModel
from upres.utils.image import Image
import uuid
import shutil


class ModelTrainer:
    """
    """

    def __init__(self, model):
        self.model = model
        self.set_up_res_model()

    def set_up_res_model(self):
        """
        This sets up a simple upscaling CNN to compare against.
        """
        up_model = keras.Sequential()
        up_model.add(
            keras.layers.UpSampling2D(
                size=(self.model.scaling, self.model.scaling),
                interpolation="bilinear",
                input_shape=(None, None, self.model.channels),
            )
        )
        self.up_model = up_model

    def train(self, images, epochs, save):
        train_images = images

        self.X = np.array([x.get_array(1 / self.model.scaling) for x in train_images])

        y = [x.get_array() for x in train_images]
        padding = int((self.model.conv_size - 1) / 2)
        y = [x[padding:-padding, padding:-padding, :] for x in y]
        self.Y = np.array(y)

        self.save_initial(images)

        for i in range(epochs):
            iteration_path = self.model.log_path / str(self.model.iteration)
            os.mkdir(iteration_path)

            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=str(iteration_path), histogram_freq=1
            )

            self.model.model.fit(
                self.X, self.Y, epochs=save, verbose=1, callbacks=[tensorboard_callback]
            )
            self.model.iteration += 1
            # self.model.model.fit(self.X, self.Y, epochs = save, verbose=1)
            print(f"Epoch {i} Loss: {self.model.model.history.history['loss'][-1]}")
            self.predict(images, self.model.iteration)

            self.model.save_model()

    def predict(self, images, step):
        x = np.array([x.get_array(1 / self.model.scaling) for x in images])
        image_names = [x.name for x in images]

        preds = self.model.model.predict(x)

        self.log_images(preds, step)

        return preds

    def save_initial(self, images):
        image_names = [x.name for x in images]

        y = [x.get_array() for x in images]
        padding = (self.model.conv_size - 1) // 2
        y = [x[padding:-padding, padding:-padding, :] for x in y]
        Y = np.array(y)

        X = np.array([x.get_array(1 / self.model.scaling) for x in images])

        up_samples = self.up_model.predict(X)
        padding = int((self.model.conv_size - 1) / 2)
        up_samples = np.array(
            [x[padding:-padding, padding:-padding, :] for x in up_samples]
        )

        self.log_images(Y, 0)
        self.log_images(up_samples, 0)

    def log_images(self, images, step):
        file_writer = tf.summary.create_file_writer(str(self.model.log_path))
        with file_writer.as_default():
            tf.summary.image(self.model.name, images / 255, max_outputs=25, step=step)

    # def plot_filters(self):

    #     weights = self.model.weights
    #     for weight in weights:
    #         for i in range( weight.shape[2]):
    #             filter = weight[:,:,i,0]
    #             plt.plot(filter)
    #             plt.show()
