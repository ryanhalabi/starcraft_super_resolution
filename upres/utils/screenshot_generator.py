from pytube import YouTube
import os
import cv2
from upres.utils.environment import env


def download_video_frames():
    yt = YouTube("https://www.youtube.com/watch?v=43u3mcs9k0Q")
    stream = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )

    stream.download(str(env.videos))

    video_files = [x for x in os.listdir(env.videos) if x != ".gitignore"]
    for video_file in video_files:
        vidcap = cv2.VideoCapture(str(env.videos / video_file))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        frame = 0
        while success:
            success, image = vidcap.read()
            frame += 1

            if (frame % (fps * 5) == 0) and (frame/fps > 65) and (frame/fps <= 2100):
                print("Time:", frame / fps)
                file_name = str(env.frames / f"{int(frame/fps)}.png")
                cv2.imwrite(file_name, image[:-250, 0:960])
