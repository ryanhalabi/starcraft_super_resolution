import boto3
import os
from pathlib import Path

from dotenv import load_dotenv

from upres import data

load_dotenv()


class Environment(object):
    def __init__(self):

        self.data_path = Path(data.__file__).parent
        self.frames = self.data_path / "input" / "frames"
        self.units = self.data_path / "input" / "units"
        self.videos = self.data_path / "input" / "videos"

        self.output = self.data_path / "output"

        self.aws_security_group_id = os.getenv("security_group_id")
        self.aws_availability_zone = os.getenv("availability_zone")
        self.aws_s3_bucket_name = os.getenv("s3_bucket_name")

        self.aws_subnet_id = os.getenv("subnet_id")
        self.aws_key_name = os.getenv("key_name")
        self.aws_vpc_id = os.getenv("vpc_id")

        s3 = boto3.client("s3", region_name=self.aws_availability_zone)
        self.s3_bucket = s3.Bucket(self.aws_s3_bucket_name)


env = Environment()
