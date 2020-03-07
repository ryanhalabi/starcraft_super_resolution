import argparse
import os

import numpy as np
from tensorflow import keras

from upres.modeling.model_trainer import ModelTrainer
from upres.modeling.sr_model import SRModel
from upres.utils.environment import env
from upres.utils.image import Image, download_unit_images
from upres.utils.screenshot_generator import download_video_frames


def download_images(units_or_frames, greyscale, scaling):
    if units_or_frames == "units":
        if len(os.listdir(env.units)) == 1:
            download_unit_images()

    elif units_or_frames == "frames":
        if len(os.listdir(env.frames)) == 1:
            download_video_frames()

    image_path = env.units if units_or_frames == "units" else env.frames

    image_files = [x for x in os.listdir(image_path) if x != ".gitignore"]
    images = [
        Image(image_path / x, greyscale=greyscale, scaling=scaling) for x in image_files
    ]

    return images


def build_keras_layers(layers, channels):
    # build middle layers
    keras_layers = []
    for layer in layers[:-1]:
        num_filters, conv_size = [int(x) for x in layer.split(",")]
        keras_layer = keras.layers.Conv2D(
            num_filters,
            (conv_size, conv_size),
            strides=(1, 1),
            padding="same",
            activation="relu",
        )

        keras_layers.append(keras_layer)

    # make last layer have appropriate channel size
    conv_size = int(layers[-1])
    keras_layer = keras.layers.Conv2D(
        channels,
        (conv_size, conv_size),
        strides=(1, 1),
        padding="same",
        activation="relu",
    )

    keras_layers.append(keras_layer)

    return keras_layers


def make_parser():
    parser = argparse.ArgumentParser(
        description="Construct and train a super resolution network"
    )
    parser.add_argument("--name", help="Name of model", default="test_model")
    parser.add_argument("--dataset", help="Units or Frames", default="units")
    parser.add_argument(
        "--layers", help="Middle layers architecture", default="1", nargs="*"
    )
    parser.add_argument("--scaling", help="Scaling of image", default=3, type=int)
    parser.add_argument(
        "--epochs", help="How many epochs to train on", default=10, type=int
    )
    parser.add_argument(
        "--epochs_per", help="How many epochs we train on before output images and save model", default=1, type=int
    )
    parser.add_argument("--greyscale", help="greyscale?", default=False, type=bool)
    parser.add_argument(
        "--overwrite", help="Whether to overwrite existing model data", default=False
    )

    return parser


# python3 run.py --h

# python3 run.py --name color_units --dataset units --layers 69,9 128,1 5 \
# --scaling 5 --epochs 1 --batches 2 --overwrite True

if __name__ == "__main__":

    parser = make_parser()
    arguments = parser.parse_args()

    dataset = arguments.dataset
    layers = arguments.layers
    name = arguments.name
    greyscale = arguments.greyscale
    scaling = arguments.scaling
    epochs = arguments.epochs
    epochs_per = arguments.epochs_per
    overwrite = False if arguments.overwrite == "False" else True

    channels = 1 if greyscale else 3

    keras_layers = build_keras_layers(layers, channels)

    # get data
    images = download_images(dataset, greyscale, scaling)
    image_shape = tuple(images[0].get_array(1 / scaling).shape)

    # build or load sr model
    sr_model = SRModel(
        name,
        image_shape,
        keras_layers,
        channels=channels,
        scaling=scaling,
        overwrite=overwrite,
    )

    mt = ModelTrainer(sr_model)
    mt.train(images, epochs=epochs, epochs_per=epochs_per)
