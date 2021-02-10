import os
import re
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
from upres.utils.environment import env
import receptive_field as rf


class SRModel:
    """
    Customizable super resolution model


    """

    def __init__(
        self,
        name,
        input_shape,
        layers,
        scaling=5,
        channels=1,
        overwrite=False,
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

        self.name = name
        self.model_path = env.output / self.name
        self.scaling = scaling
        self.channels = channels
        self.conv_size = 2 * self.scaling - 1
        self.input_shape = input_shape
        self.layers = layers

        self.set_optimizer()

        if overwrite and os.path.isdir(self.model_path):
            # remove locally model data
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
            self.start_epoch = 0

            if not os.path.isdir(self.model_path):
                os.mkdir(self.model_path)
                os.mkdir(self.model_path / "models")
                os.mkdir(self.model_path / "logs")
                os.mkdir(self.model_path / "images")
                os.mkdir(self.model_path / "images" / "static")

        self.log_path = self.model_path / "logs"
        self.images_path = self.model_path / "images"

    def load_model(self, model_files):

        epoch = max([int(re.findall(r"_(\d+).hdf5", x)[0]) for x in model_files])
        model_file_path = self.model_path / "models" / f"{self.name}_{epoch}.hdf5"

        self.start_epoch = epoch + 1
        print(f"Loading model {self.name}_{epoch}.hdf5")
        self.model = keras.models.load_model(str(model_file_path))

    def make_model(self):

        self.calculate_receptive_field()

        inputs = keras.layers.Input(
            shape=(self.input_shape[0], self.input_shape[1], self.channels),
            name="input",
        )

        # upscaler
        upscaler = keras.layers.Conv2DTranspose(
            self.channels,
            (self.conv_size, self.conv_size),
            strides=(self.scaling, self.scaling),
            padding="same",
            name="upscaler",
        )(inputs)

        # trainable parameters
        keras_layers = self.build_keras_layers()
        last_layer = self.apply_layers(keras_layers, upscaler)

        # combine upscale + trainable parameters, we predict residual
        add_layer = keras.layers.add(
            [last_layer, upscaler], name="residual_plus_upscaler"
        )

        cropping_layer = self.make_cropping_layer()
        predictions = cropping_layer(add_layer)

        # finalize model
        model = keras.Model(inputs=inputs, outputs=predictions)
        model.compile(self.optimizer, "mean_squared_error")

        print(f"Created new model: {self.name}")

        return model

    def apply_layers(self, keras_layers, input):

        first_layer = keras_layers[0]
        previous_layer = first_layer(input)

        for layer in keras_layers[1:]:
            previous_layer = layer(previous_layer)

        return previous_layer

    def make_cropping_layer(self):
        """
        Creates layer that crops out points whose receptive field falls out of bounds.
        """

        # the points to crop out from the transpose convolutional layer
        # conv_transpose_crop = (self.conv_size - self.scaling) / 2

        # the points to crop out from since they absorb ghost convolutional points
        cropping_layer = keras.layers.Cropping2D(cropping=self.rf_padding)

        return cropping_layer

    def set_optimizer(self):
        self.optimizer = keras.optimizers.Adam()

    def build_keras_layers(self):
        # build middle layers
        keras_layers = []
        for i, layer in enumerate(self.layers[:-1]):
            num_filters, conv_size = [int(x) for x in layer.split(",")]
            keras_layer = keras.layers.Conv2D(
                num_filters,
                (conv_size, conv_size),
                strides=(1, 1),
                padding="same",
                activation="relu",
                name=f"l_{i}",
            )

            keras_layers.append(keras_layer)

        # make last layer have appropriate channel size
        conv_size = int(self.layers[-1])
        keras_layer = keras.layers.Conv2D(
            self.channels,
            (conv_size, conv_size),
            strides=(1, 1),
            padding="same",
            name=f"l_{len(self.layers)-1}",
        )

        keras_layers.append(keras_layer)

        assert all(
            [x.kernel_size[0] % 2 != 0 for x in keras_layers]
        ), "All kernels must be odd sized"

        return keras_layers

    def calculate_receptive_field(self):
        g = tf.Graph()
        with g.as_default():

            inputs = keras.layers.Input(
                shape=(1000, 1000, self.channels),
                name="input",
            )

            keras_layers = self.build_keras_layers()
            last_layer = self.apply_layers(keras_layers, inputs)

            model = keras.Model(inputs=inputs, outputs=last_layer)

        receptive_field = rf.compute_receptive_field_from_graph_def(
            g, "input", model.outputs[0].op.name
        )
        self.rf_stride = int(receptive_field.stride[0])
        self.rf_size = int(receptive_field.size[0])
        self.rf_padding = int(receptive_field.padding[0])
