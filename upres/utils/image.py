import os
import re
from pathlib import Path

import cv2
import numpy as np
import requests

from upres.utils.environment import env


class Image:
    def __init__(self, path: Path, greyscale: bool, scaling: int):
        self.path = path
        self.greyscale = greyscale
        self.read_type = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR
        self.scaling = scaling

    @property
    def name(self):
        return re.search(r"/([\w]*).png", str(self.path)).group(1)

    def get_array(self, scale=1):
        array = cv2.imread(str(self.path), self.read_type)

        # resize original image so it can be be scaled without fractions
        x_extra = array.shape[0] % self.scaling
        y_extra = array.shape[1] % self.scaling

        x_extra = self.scaling - x_extra if x_extra != 0 else x_extra
        y_extra = self.scaling - y_extra if y_extra != 0 else y_extra

        padded_array = cv2.resize(
            array, (int(array.shape[1] + x_extra), int(array.shape[0] + y_extra))
        )

        # scale image
        resized_array = cv2.resize(
            padded_array,
            (int(padded_array.shape[1] * scale), int(padded_array.shape[0] * scale)),
        )

        # cv2 reads in array as BGR, tensorboard shows as RGB
        x = np.copy(resized_array)
        resized_array[:, :, 0] = x[:, :, 2]
        resized_array[:, :, 2] = x[:, :, 0]

        # cv2.imshow('image',array)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if self.greyscale:
            resized_array = np.expand_dims(resized_array, 2)
        return resized_array


def download_images(urls):

    for url in urls:
        # unit = re.search(r"[/\d]([A-Za-z_]*).png", url).group(1)
        unit = re.search(r"[/\d]([\w]*).jpg", url).group(1)
       file_name = str(env.units / f"{unit}.png")
        with open(file_name, "wb+") as f:
            f.write(requests.get(url).content)

            print(Image(file_name, False, 1).get_array().shape)


def download_unit_images():
    image_urls = [
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/UXAGLN2RQ8001492473801945.jpg",
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/MOM5V4N13UVI1492476408599.jpg",
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/MBKXXJZ6YCVV1492476407508.jpg",
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/TA17UJIR1TV31492476406352.jpg",
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/CRSJ1ZI8GNQA1492476407623.jpg",
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/CMQOBEWCGCUE1492476407691.jpg",
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/5VKFTLLMYUUH1492476407602.jpg",
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/HJIMRH145ATT1492476406520.jpg",
        "https://bnetcmsus-a.akamaihd.net/cms/gallery/0AAPN0GIQ0I91492476406344.jpg",
        # "https://liquipedia.net/commons/images/9/9d/Marine.png",
        # "https://liquipedia.net/commons/images/8/8a/Firebat.png",
        # "https://liquipedia.net/commons/images/2/26/Medic.png",
        # "https://liquipedia.net/commons/images/f/f7/Scv.png",
        # "https://liquipedia.net/commons/images/a/ab/Ghost.png",
    ]
    download_images(image_urls)


def download_building_images():
    image_urls = [
        "https://liquipedia.net/commons/images/thumb/1/13/%24Academy.png/600px-%24Academy.png",
        "https://liquipedia.net/commons/images/thumb/d/dd/%24Armory.png/600px-%24Armory.png",
        "https://liquipedia.net/commons/images/thumb/d/df/%24Barracks.png/600px-%24Barracks.png",
        "https://liquipedia.net/commons/images/thumb/e/e9/%24Bunker.png/600px-%24Bunker.png",
        "https://liquipedia.net/commons/images/thumb/d/dc/%24Command_Center.png/600px-%24Command_Center.png",
        "https://liquipedia.net/commons/images/thumb/1/1f/%24Comsat_Station.png/600px-%24Comsat_Station.png",
        "https://liquipedia.net/commons/images/thumb/2/2b/%24Control_Tower.png/600px-%24Control_Tower.png",
        "https://liquipedia.net/commons/images/thumb/0/04/%24Covert_Ops.png/600px-%24Covert_Ops.png",
        "https://liquipedia.net/commons/images/thumb/4/41/%24Engineering_Bay.png/600px-%24Engineering_Bay.png",
        "https://liquipedia.net/commons/images/thumb/3/36/%24Factory.png/600px-%24Factory.png",
        "https://liquipedia.net/commons/images/thumb/0/0b/%24Machine_Shop.png/600px-%24Machine_Shop.png",
        "https://liquipedia.net/commons/images/thumb/4/41/%24Missile_Turret.png/600px-%24Missile_Turret.png",
        "https://liquipedia.net/commons/images/thumb/e/ed/%24Nuclear_Silo.png/600px-%24Nuclear_Silo.png",
        "https://liquipedia.net/commons/images/thumb/7/7a/%24Physics_Lab.png/600px-%24Physics_Lab.png",
        "https://liquipedia.net/commons/images/thumb/c/ce/%24Refinery.png/600px-%24Refinery.png",
        "https://liquipedia.net/commons/images/thumb/2/25/%24Science_Facility.png/600px-%24Science_Facility.png",
        "https://liquipedia.net/commons/images/thumb/2/24/%24Starport.png/600px-%24Starport.png",
        "https://liquipedia.net/commons/images/thumb/c/c7/%24Supply_Depot.png/600px-%24Supply_Depot.png",
    ]
    download_images(image_urls)
