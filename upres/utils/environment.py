from pathlib import Path
import os

from upres import data

class Environment(object):
    def __init__(self):

        pwd = data.__file__
        pwd = Path(pwd.split('/__init__.py')[0])

        self.frames = pwd / "input" / "frames"
        self.units = pwd / "input" / "units"
        self.videos = pwd / "input" / "videos"

        self.output = pwd / "output"


env = Environment()
