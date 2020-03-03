import os

import keras
import numpy as np

from upres.modeling.model_trainer import ModelTrainer
from upres.modeling.sr_model import SRModel
from upres.utils.environment import env
from upres.utils.image import Image, download_unit_images
from upres.utils.screenshot_generator import download_video_frames

# download unit images if empty
if len(os.listdir(env.units)) == 1:
    download_unit_images()

# download frame images if empty
# if len(os.listdir(env.frames)) == 1:
#     download_video_frames()

scaling = 5
greyscale = False
channels = 1 if greyscale else 3


frame_files = [x for x in os.listdir(env.frames) if x != ".gitignore"]
frame_images = [
    Image(env.frames / x, greyscale=greyscale, scaling=scaling) for x in frame_files
]
frame_shape = tuple(frame_images[0].get_array(1 / scaling).shape)


unit_files = [x for x in os.listdir(env.units) if x != ".gitignore"]
unit_images = [
    Image(env.units / x, greyscale=greyscale, scaling=scaling) for x in unit_files
]
unit_shape = tuple(unit_images[0].get_array(1 / scaling).shape)


# FRAMES

layers = [
    keras.layers.Conv2D(32, (9, 9), strides=(1, 1), padding="same", activation="relu"),
    keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation="relu"),
    keras.layers.Conv2D(channels, (5, 5), strides=(1, 1), padding="same"),
]

# sr_model = SRModel(
#     "color",
#     frame_shape,
#     layers,
#     channels=channels,
#     scaling=scaling,
#     overwrite=False,
# )

# mt = ModelTrainer(sr_model)
# # mt.train(images, epochs=0, batches=0, log=False)
# mt.train(images, epochs=50, batches=10000)


# UNITS

layers = [
    keras.layers.Conv2D(32, (9, 9), strides=(1, 1), padding="same", activation="relu"),
    keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation="relu"),
    keras.layers.Conv2D(channels, (5, 5), strides=(1, 1), padding="same"),
]

sr_model = SRModel(
    "color", unit_shape, layers, channels=channels, scaling=scaling, overwrite=False,
)

mt = ModelTrainer(sr_model)
mt.train(unit_images, epochs=50, batches=10000)
