import numpy as np
import requests
import re
from environment import env

class DownloadImages:
    def __init__(self, urls):
        self.urls = urls

    def download_images(self):
        for url in self.urls:
            unit = re.search( r"[/\d]([\w]*).png", url).group(1)
            file_name = env.path + f"/source/{unit}.png"
            with open(file_name,'wb+') as f:
                f.write(requests.get(url).content)


