import os
import subprocess
from pathlib import Path

import boto3
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
        self.aws_access_key_id = os.getenv("aws_access_key_id")
        self.aws_secret_access_key = os.getenv("aws_secret_access_key")
        self.aws_s3_bucket_name = os.getenv("s3_bucket_name")
        self.aws_s3_bucket_uri = f"s3://{self.aws_s3_bucket_name}"

        self.aws_subnet_id = os.getenv("subnet_id")
        self.aws_key_name = os.getenv("key_name")
        self.aws_vpc_id = os.getenv("vpc_id")

    def sync_with_s3(self):
        """
        Runs bash command to sync local files with s3 storage.
        """
        sync_bash_command = [
            "aws",
            "s3",
            "sync",
            self.data_path,
            self.aws_s3_bucket_uri,
            "--only-show-errors"
        ]

        subprocess.call(sync_bash_command)


env = Environment()
