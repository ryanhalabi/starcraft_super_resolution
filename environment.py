import os
from pathlib import Path

class Environment(object):

    @property
    def path(self):
        return Path(os.getcwd()) 

env = Environment()