import os
import re
import shutil

import numpy as np
from tensorflow import keras

from upres.utils.environment import env


class SRModel:
    """
    Customizable super resolution model


    """

    def __init__(
        self, name, input_shape, layers, scaling=5, channels=1, overwrite=False,
    ):
        """
        Parameters
        ----------
        name : str
            Name of model, used when saving outputs.
        input_shape : (width, height)
            Shape of inputs, used when building input layer.
        layers : [keras.Layer]
            List of Keras layers used in middle of network.
        scaling : odd int
            Downscaling to apply to images before attempting to recreation.
        channels : int, 3 or 1
            Number of output channels, color = 3, greyscale = 1.
        overwrite : bool
            If True, remove all data for model of given name and start fresh.
            If False, will attempt to load.
        """

        assert scaling % 2 != 0, "Scaling factor must be odd"
        assert all([x.kernel_size[0] % 2 != 0 for x in self.layers]), "All kernels must be odd sized"

        self.name = name
        self.model_path = env.output / self.name
        self.scaling = scaling
        self.channels = channels
        self.conv_size = 2 * self.scaling - 1
        self.input_shape = input_shape
        self.layers = layers

        self.max_kernel_size = max([x.kernel_size[0] for x in self.layers])
        self.set_optimizer()

        if overwrite and os.path.isdir(self.model_path):
            shutil.rmtree(self.model_path)
            print(f"Removed existing {self.name} model data.")

        self.create_or_load_model()

    def create_or_load_model(self):

        model_files = (
            os.listdir(self.model_path / "models")
            if os.path.isdir(self.model_path / "models")
            else None
        )
        if model_files:
            self.load_model(model_files)
        else:

            self.model = self.make_model()
            self.iteration = 0

            if not os.path.isdir(self.model_path):
                os.mkdir(self.model_path)
                os.mkdir(self.model_path / "models")
                os.mkdir(self.model_path / "logs")
                os.mkdir(self.model_path / "images")
            self.save_model()

        self.log_path = self.model_path / "logs"
        self.images_path = self.model_path / "images"

    def load_model(self, model_files):

        iteration = max([int(re.findall(r"_(\d+).hdf5", x)[0]) for x in model_files])
        model_file_path = self.model_path / "models" / f"{self.name}_{iteration}.hdf5"

        self.iteration = iteration
        print(f"Loading model {self.name}_{iteration}.hdf5")
        self.model = keras.models.load_model(str(model_file_path))

    def make_model(self):

        inputs = keras.layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], self.channels)
        )

        # upscaler
        upscaler = keras.layers.Conv2DTranspose(
            self.channels,
            (self.conv_size, self.conv_size),
            strides=(self.scaling, self.scaling),
        )(inputs)

        # trainable paramters
        last_layer = self.apply_layers(upscaler)

        # combine upscale + trainable parameters, we predict residual
        add_layer = keras.layers.add([last_layer, upscaler])

        depad_layer = self.make_depad_layer()
        predictions = depad_layer(add_layer)

        # finalize model
        model = keras.Model(inputs=inputs, outputs=predictions)
        model.compile(self.optimizer, "mean_squared_error")

        print(f"Created new model: {self.name}")

        return model

    def apply_layers(self, input):

        first_layer = self.layers[0]
        previous_layer = first_layer(input)

        for layer in self.layers[1:]:
            previous_layer = layer(previous_layer)

        return previous_layer

    def make_depad_layer(self):
        """
        Removes ghost points from conv2d and points that are influenced by ghost points from convolutions.

        self.conv_size - self.scaling + 1: removes the ghost points from conv2d 
        self.max_kernel_size - 1: removes any points that are influenced by boundary bad points from conv2d
        """
        depad_filter_size = (self.conv_size - self.scaling + 1) + (
            self.max_kernel_size - 1
        )
        depad_kernel = np.zeros(
            [depad_filter_size, depad_filter_size, self.channels, self.channels]
        )
        center = int(depad_kernel.shape[0] / 2)
        for i in range(self.channels):
            depad_kernel[center, center, i, i] = 1

        depad_bias = np.zeros([self.channels])

        depad_layer = keras.layers.Conv2D(
            self.channels,
            (depad_filter_size, depad_filter_size),
            strides=(1, 1),
            weights=[depad_kernel, depad_bias],
            trainable=False,
        )

        return depad_layer

    def set_optimizer(self):
        self.optimizer = keras.optimizers.Adam()

    def save_model(self):
        self.model.save(
            str(self.model_path / "models" / f"{self.name}_{self.iteration}.hdf5")
        )
