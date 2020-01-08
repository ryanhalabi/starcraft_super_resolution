from pytube import YouTube
import os
import cv2

video_folder = '/Users/ryan/starcraft_upres/videos/'
yt = YouTube('https://www.youtube.com/watch?v=lTyco9kbFdo')
stream = yt.streams.filter(progressive=True, file_extension='mp4')\
   .order_by('resolution').desc().first()

stream.download(video_folder)

for file in os.listdir(video_folder):
    vidcap = cv2.VideoCapture(video_folder + file)
    fps = vidcap.get(cv2.CAP_PROP_FPS) 
    success,image = vidcap.read()
    frame = 0
    while success:
        success,image = vidcap.read()
        frame += 1

        if frame%(fps*5) == 0:
            print('Time:', frame/fps)
            cv2.imwrite(f"frames/{int(frame/fps)}.png", image[:-250,0:960])

