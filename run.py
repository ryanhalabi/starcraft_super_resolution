from download_images import DownloadImages
from image import Image
from model import TrainSRModel, Model
from environment import env
import os
import numpy as np

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

DownloadImages(image_urls)


images = [Image( env.path / 'data' / 'source' / x, greyscale=True) for x in os.listdir(env.path / 'data' / "source")]
model = Model('test',channels=1)

mt = TrainSRModel(model, False)


x = np.array([x.get_array(mt.model.scaling) for x in images])
image_names = [x.name for x in images]

preds = mt.model.model.predict(x)


mt.train(images, 500, 500)


# import matplotlib.pyplot as plt
# plt.matshow(mt.model.model.layers[0].get_weights()[0][:,:,0,0])
# plt.show()


# import numpy as np
# x = images[0].get_array(4)
# y = mt.model.predict( np.array([x]))
# # cv2.imwrite( str(env.path / 'images' / f"Atest.png"), x)


