import argparse
import os

import numpy as np
from tensorflow import keras

from upres.modeling.model_trainer import ModelTrainer
from upres.modeling.sr_model import SRModel
from upres.utils.environment import env
from upres.utils.image import Image, download_unit_images

# from upres.utils.screenshot_generator import download_video_frames


def download_images(units_or_frames, greyscale, scaling):
    if units_or_frames == "units":
        if len(os.listdir(env.units)) == 1:
            download_unit_images()

    # elif units_or_frames == "frames":
    #     if len(os.listdir(env.frames)) == 1:
    #         download_video_frames()

    image_path = env.units if units_or_frames == "units" else env.frames
    image_files = [x for x in os.listdir(image_path) if x != ".gitignore"]
    image_files = sorted(image_files)
    images = [
        Image(image_path / x, greyscale=greyscale, scaling=scaling) for x in image_files
    ]

    return images


def make_parser():
    parser = argparse.ArgumentParser(
        description="Construct and train a super resolution network"
    )
    parser.add_argument("--name", help="Name of model", default="test_model")
    parser.add_argument("--dataset", help="Units or Frames", default="units")
    parser.add_argument(
        "--layers", help="Middle layers architecture", default="1", nargs="*"
    )
    parser.add_argument("--loss", help="MSE or GAN", default="MSE")
    parser.add_argument("--scaling", help="Scaling of image", default=3, type=int)
    parser.add_argument(
        "--epochs", help="How many epochs to train on", default=10, type=int
    )
    parser.add_argument(
        "--batch_size", help="Training batch size", default=32, type=int
    )
    parser.add_argument(
        "--epochs_per_save",
        help="How many epochs we train on before output images and save model",
        default=1,
        type=int,
    )
    parser.add_argument("--greyscale", help="greyscale?", default="False")
    parser.add_argument("--s3_sync", help="sync with s3", default="False")
    parser.add_argument(
        "--overwrite", help="Whether to overwrite existing model data", default="False"
    )

    return parser


# python3 /starcraft_super_resolution/run.py --name a_non_relu --dataset units --layers 128,9 256,1 19 --scaling 3 --epochs 20000000000 --batch_size 32 --epochs_per_save 100 --greyscale False --overwrite False
# python3 run.py --name local_test --dataset units --layers 128,9 256,1 19 --scaling 3 --epochs 5 --overwrite True --s3_sync False --loss GAN
# tensorboard --logdir=/Users/ryan/projects/starcraft_super_resolution/upres/data/output --port=8080  --bind_all --max_reload_threads 1 --samples_per_plugin='images=200'

if __name__ == "__main__":

    parser = make_parser()
    arguments = parser.parse_args()

    dataset = arguments.dataset
    layers = arguments.layers
    name = arguments.name
    loss = arguments.loss
    greyscale = False if arguments.greyscale == "False" else True
    scaling = arguments.scaling
    epochs = arguments.epochs
    batch_size = arguments.batch_size
    epochs_per_save = arguments.epochs_per_save
    s3_sync = False if arguments.s3_sync == "False" else True
    overwrite = False if arguments.overwrite == "False" else True

    channels = 1 if greyscale else 3

    # get data
    images = download_images(dataset, greyscale, scaling)

    image_shape = tuple(images[0].get_array(1 / scaling).shape)
    name = f"{name}_{dataset}_{layers}_{loss}"

    # build or load sr model
    sr_model = SRModel(
        name,
        image_shape,
        layers,
        loss,
        channels=channels,
        scaling=scaling,
        overwrite=overwrite,
    )

    mt = ModelTrainer(sr_model)
    mt.train(
        images,
        epochs=epochs,
        batch_size=batch_size,
        epochs_per_save=epochs_per_save,
        s3_sync=s3_sync,
    )
