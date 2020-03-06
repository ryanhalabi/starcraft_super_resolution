import os
from pathlib import Path

from dotenv import load_dotenv
from upres import data

load_dotenv()


class Environment(object):
    def __init__(self):
        pwd = data.__file__
        pwd = Path(pwd.split("/__init__.py")[0])

        self.frames = pwd / "input" / "frames"
        self.units = pwd / "input" / "units"
        self.videos = pwd / "input" / "videos"

        self.output = pwd / "output"

        self.aws_security_group_id = os.getenv("security_group_id")
        self.aws_subnet_id = os.getenv("subnet_id")
        self.aws_availability_zone = os.getenv("availability_zone")
        self.aws_key_name = os.getenv("key_name")
        self.aws_vpc_id = os.getenv("vpc_id")


env = Environment()
