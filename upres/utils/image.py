import cv2
import numpy as np
import os
import re
import requests
from upres.utils.environment import env
from pathlib import Path


class Image:
    def __init__(self, path: Path, greyscale: bool):
        self.path = path
        self.greyscale = greyscale
        self.read_type = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR

    @property
    def name(self):
        return re.search(r"/([\w]*).png", str(self.path)).group(1)

    def get_array(self, scale=1):
        array = cv2.imread(str(self.path), self.read_type)
        array = cv2.resize(array, (int(array.shape[1] * scale), int(array.shape[0] * scale)))

        # cv2.imshow('image',array)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        if self.greyscale:
            array = np.expand_dims(array, 2)
        return array


def download_images(urls):

    for url in urls:
        unit = re.search(r"[/\d]([\w]*).png", url).group(1)
        file_name = str(env.frames / f"{unit}.png")
        with open(file_name, "wb+") as f:
            f.write(requests.get(url).content)


def download_unit_images():
    image_urls = [
        "https://liquipedia.net/commons/images/9/9d/Marine.png",
        "https://liquipedia.net/commons/images/8/8a/Firebat.png",
        "https://liquipedia.net/commons/images/2/26/Medic.png",
        "https://liquipedia.net/commons/images/f/f7/Scv.png",
        "https://liquipedia.net/commons/images/a/ab/Ghost.png",
        # "https://liquipedia.net/commons/images/thumb/1/13/%24Academy.png/600px-%24Academy.png",
        # "https://liquipedia.net/commons/images/thumb/d/dd/%24Armory.png/600px-%24Armory.png",
        # "https://liquipedia.net/commons/images/thumb/d/df/%24Barracks.png/600px-%24Barracks.png",
        # "https://liquipedia.net/commons/images/thumb/e/e9/%24Bunker.png/600px-%24Bunker.png",
        # "https://liquipedia.net/commons/images/thumb/d/dc/%24Command_Center.png/600px-%24Command_Center.png",
        # "https://liquipedia.net/commons/images/thumb/1/1f/%24Comsat_Station.png/600px-%24Comsat_Station.png",
        # "https://liquipedia.net/commons/images/thumb/2/2b/%24Control_Tower.png/600px-%24Control_Tower.png",
        # "https://liquipedia.net/commons/images/thumb/0/04/%24Covert_Ops.png/600px-%24Covert_Ops.png",
        # "https://liquipedia.net/commons/images/thumb/4/41/%24Engineering_Bay.png/600px-%24Engineering_Bay.png",
        # "https://liquipedia.net/commons/images/thumb/3/36/%24Factory.png/600px-%24Factory.png",
        # "https://liquipedia.net/commons/images/thumb/0/0b/%24Machine_Shop.png/600px-%24Machine_Shop.png",
        # "https://liquipedia.net/commons/images/thumb/4/41/%24Missile_Turret.png/600px-%24Missile_Turret.png",
        # "https://liquipedia.net/commons/images/thumb/e/ed/%24Nuclear_Silo.png/600px-%24Nuclear_Silo.png",
        # "https://liquipedia.net/commons/images/thumb/7/7a/%24Physics_Lab.png/600px-%24Physics_Lab.png",
        # "https://liquipedia.net/commons/images/thumb/c/ce/%24Refinery.png/600px-%24Refinery.png",
        # "https://liquipedia.net/commons/images/thumb/2/25/%24Science_Facility.png/600px-%24Science_Facility.png",
        # "https://liquipedia.net/commons/images/thumb/2/24/%24Starport.png/600px-%24Starport.png",
        # "https://liquipedia.net/commons/images/thumb/c/c7/%24Supply_Depot.png/600px-%24Supply_Depot.png"
    ]
    download_images(image_urls)
