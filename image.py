import cv2
import numpy as np
import os
import re 

class Image:

    def __init__(self, path, greyscale):
        self.path = path
        self.greyscale = greyscale
        self.read_type = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR

    @property
    def name(self):
        return re.search( r"/([\w]*).png", str(self.path)).group(1)

    def get_array(self, scale=1):
        array = cv2.imread(str(self.path) , self.read_type)
        if scale != 1:
            array = cv2.resize(array, (int(array.shape[0]/scale), int(array.shape[1]/scale)))
        if self.greyscale:
            array = np.expand_dims(array,2)
        return array
