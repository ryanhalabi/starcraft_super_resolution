"""
Used to generate gifs from static image outputs
"""

from PIL import Image, ImageDraw
import os

static_image_folder = "/Users/ryan/projects/starcraft_super_resolution/upres/data/output/a_units_['128,9', '256,1', '19']/images/static"

images = os.listdir(static_image_folder)
image_names = {x.split("_")[0] for x in images}

for image_name in image_names:
    named_images = {
        int(x.split("_")[-1][:-4]): x for x in images if x.split("_")[0] == image_name
    }
    ordered_images = [named_images[x] for x in sorted(named_images.keys())]

    hi_res, base, *imgs = [
        Image.open(f"{static_image_folder}/{x}") for x in ordered_images
    ]
    fp_out = f"{image_name}.gif"

    width = hi_res._size[0] + 10
    height = hi_res._size[1]
    new_imgs = []
    for i, img in enumerate(imgs):
        new_im = Image.new("RGB", (width * 3, height))

        new_im.paste(base, (0, 0))
        new_im.paste(img, (width, 0))
        new_im.paste(hi_res, (2 * width, 0))

        draw = ImageDraw.Draw(new_im)

        draw.text((width / 2, 10), "bilinear interpolation")
        draw.text((3 * width / 2, 10), f"epoch={i*100}")
        draw.text((5 * width / 2, 10), "target")

        new_imgs.append(new_im)

    new_imgs[0].save(
        fp=f"{static_image_folder}/{fp_out}",
        format="GIF",
        append_images=new_imgs[1:],
        save_all=True,
        duration=700,
        loop=0,
    )


# # filepaths
# static_image_folder = "/Users/ryan/projects/starcraft_super_resolution/upres/data/output/a_units_['128,9', '256,1', '19']/images/static"
# images = os.listdir(static_image_folder)
# image_names = list({x.split("_")[0] for x in images})

# first_image = Image.open(f"{static_image_folder}/{images[0]}")
# image_times = sorted(
#     [int(x.split("_")[-1][:-4]) for x in images if x.split("_")[0] == image_names[0]]
# )

# width = first_image._size[0] + 10
# height = first_image._size[1] + 10


# new_imgs = []
# for time in image_times[2:]:
#     print(time)

#     new_im = Image.new("RGB", (width * 3, len(image_names) * height))

#     for i, name in enumerate(image_names):

#         base = Image.open(f"{static_image_folder}/{name}_-1.jpg")
#         hi_res = Image.open(f"{static_image_folder}/{name}_-2.jpg")

#         img = Image.open(f"{static_image_folder}/{name}_{time}.jpg")

#         new_im.paste(base, (0, i * height))
#         new_im.paste(img, (width, i * height))
#         new_im.paste(hi_res, (2 * width, i * height))

#         draw = ImageDraw.Draw(new_im)

#         draw.text((width / 2, 10 + i * height), "input")
#         draw.text((3 * width / 2, 10 + i * height), f"epoch={time}")
#         draw.text((5 * width / 2, 10 + i * height), f"target")

#         new_imgs.append(new_im)


# fp_out = f"mega_gif.gif"

# new_imgs[0].save(
#     fp=fp_out,
#     format="GIF",
#     append_images=new_imgs[1:],
#     save_all=True,
#     duration=200,
#     loop=0,
# )
