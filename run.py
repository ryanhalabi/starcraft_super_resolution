from upres.utils.image import Image, download_unit_images
from upres.modeling.model_trainer import ModelTrainer 
from upres.modeling.sr_model import SRModel
from upres.utils.environment import env
from upres.utils.image import download_unit_images
import os
import numpy as np


#download images if empty
if len(os.listdir(env.units)) == 1:
    download_unit_images()

#COLOR
greyscale = False
channels = 1 if greyscale else 3
scaling = 2

images = [Image( env.frames / x, greyscale=greyscale) for x in os.listdir(env.units)]
image_shape = tuple(images[0].get_array(1 / scaling).shape)
assert [x/scaling == int(x/scaling) for x in image_shape]

sr_model = SRModel('color', image_shape, channels=channels, scaling=scaling, conv_size=5, overwrite=False)

mt = ModelTrainer(sr_model)
# mt.train(images, epochs=0, batches=0, log=False)
mt.train(images, epochs=1, batches=200)



# #GREYSCALE
# greyscale = True
# channels = 1 if greyscale else 3
# scaling = 2

# images = [Image( env.frames / x, greyscale=greyscale) for x in os.listdir(env.frames)][0:1]
# image_shape = tuple(images[0].get_array(1 / scaling).shape)
# assert [x/scaling == int(x/scaling) for x in image_shape]

# sr_model = SRModel('grey', image_shape, channels=channels, scaling=scaling, conv_size=5, overwrite=True)

# mt = ModelTrainer(sr_model)
# mt.train(images, epochs=10, batches=30)


# x = np.array([x.get_array(scale_factor) for x in images])
# image_names = [x.name for x in images]








# import matplotlib.pyplot as plt
# plt.matshow(mt.model.model.layers[0].get_weights()[0][:,:,0,0])
# plt.show()


# import numpy as np
# x = images[0].get_array(4)
# y = mt.model.predict( np.array([x]))
# # cv2.imwrite( str(env.path / 'images' / f"Atest.png"), x)






# import matplotlib.pyplot as plt
# preds = mt.predict(images,'test')

# print(preds.shape)
# x = mt.predict(images, image_names)

# plt.imshow(x[0,:,:,:])


# for j, pred in enumerate(preds):
#   cv2.imwrite( str(env.path / 'data' / 'test_color' / f"{image_names[j]}.png"), y)


# mt.model.model.history.history['loss'][-1]


# input = keras.layers.Input(shape=(5, 5, 3))

# depad_filter_size = 3
# depad_kernel = np.zeros([depad_filter_size, depad_filter_size, 3, 3])
# center = int(depad_kernel.shape[0]/2)

# for i in range(3):
#   depad_kernel[center, center,i,i] = 1

# depad_bias = np.zeros([3])

# out = keras.layers.Conv2D(3, (depad_filter_size, depad_filter_size), strides=(1,1),
#                                 weights=[depad_kernel, depad_bias], trainable=False)(input)


# model = keras.Model(inputs=input, outputs=out)

# model.compile(keras.optimizers.Adam(), 'mean_squared_error')

# model.summary()

# x = np.random.randn(1,5,5,3)
# y = model.predict(x)
# print((y - x[:,1:-1,1:-1,:]).sum())

# print(x[0,:,:,0])
# print(y[0,:,:,0])

# print(x[0,:,:,1])
# print(y[0,:,:,1])

