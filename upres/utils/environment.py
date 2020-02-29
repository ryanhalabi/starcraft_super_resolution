from pathlib import Path
import os


class Environment(object):
    def __init__(self):
        pwd = Path(os.getcwd())

        self.frames = pwd / "data" / "input_data" / "frames"
        self.units = pwd / "data" / "input_data" / "units"
        self.output_data = pwd / "data" / "output_data"


env = Environment()
