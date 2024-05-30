import numpy
import cv2
from moviepy.editor import *
from moviepy.config import change_settings
from pydub import AudioSegment 
def rect_with_rounded_corners(image, r, t, c):
    c += (255, )
    h, w = image.shape[:2]
    new_image = numpy.ones((h+2*t, w+2*t, 4), numpy.uint8) * 255
    new_image[:, :, 3] = 0
    new_image = cv2.ellipse(new_image, (int(r+t/2), int(r+t/2)), (r, r), 180, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(r+t/2)), (r, r), 270, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(r+t/2), int(h-r+3*t/2-1)), (r, r), 90, 0, 90, c, t)
    new_image = cv2.ellipse(new_image, (int(w-r+3*t/2-1), int(h-r+3*t/2-1)), (r, r), 0, 0, 90, c, t)
    new_image = cv2.line(new_image, (int(r+t/2), int(t/2)), (int(w-r+3*t/2-1), int(t/2)), c, t)
    new_image = cv2.line(new_image, (int(t/2), int(r+t/2)), (int(t/2), int(h-r+3*t/2)), c, t)
    new_image = cv2.line(new_image, (int(r+t/2), int(h+3*t/2)), (int(w-r+3*t/2-1), int(h+3*t/2)), c, t)
    new_image = cv2.line(new_image, (int(w+3*t/2), int(r+t/2)), (int(w+3*t/2), int(h-r+3*t/2)), c, t)
    mask = new_image[:, :, 3].copy()
    mask = cv2.floodFill(mask, None, (int(w/2+t), int(h/2+t)), 128)[1]
    mask[mask != 128] = 0
    mask[mask == 128] = 1
    mask = numpy.stack((mask, mask, mask), axis=2)
    temp = numpy.zeros_like(new_image[:, :, :3])
    temp[(t-1):(h+t-1), (t-1):(w+t-1)] = image.copy()
    new_image[:, :, :3] = new_image[:, :, :3] * (1 - mask) + temp * mask
    temp = new_image[:, :, 3].copy()
    new_image[:, :, 3] = cv2.floodFill(temp, None, (int(w/2+t), int(h/2+t)), 255)[1]
    return new_image

def convert_frames_to_video(time,devil_image="angel.png",pathOut='video.mp4',fps=2.0):
    frame_array,img=[],cv2.imread(devil_image)
    files = [devil_image]*(time*int(fps))
    for i in range(len(files)):
        filename=files[i]
        height,width,layers =img.shape
        size=(width,height)
        frame_array.append(img)
    out=cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    del frame_array,files,filename,height,width,layers,size,out
        
cv2.imwrite("angel.png",numpy.array([255]*(1080*720*3)).reshape((1080,720,3)))
convert_frames_to_video(2)
frame_devil,x,devil_count,y=[],150,0,590
clip,devil=VideoFileClip("video.mp4"),cv2.resize(rect_with_rounded_corners(cv2.imread("demon.jpg"),10,1,(255,255,255)),(500,500))
devil_up=TextClip(f"——————————",fontsize = 720//15,color = 'black').set_pos(('center',770))

for i in range(5):
    devil_up_1=TextClip(f"•",fontsize = 720//20,color = 'black').set_pos((y,780))
    if i%2==0:
        for i in range(10):
            image=ImageClip(devil).set_position(("center",x))
            CompositeVideoClip([clip,image,devil_up,devil_up_1]).save_frame(f"frame_devil/{devil_count}.png",t=1)
            x+=1
            devil_count+=1
    else:
        for i in range(10):
            image=ImageClip(devil).set_position(("center",x))
            CompositeVideoClip([clip,image,devil_up,devil_up_1]).save_frame(f"frame_devil/{devil_count}.png",t=1)
            x-=1
            devil_count+=1
    image.close()

video_=concatenate_videoclips([ImageClip(f"frame_devil/{i}.png").set_duration(0.10) for i in range(devil_count)])
video_.write_videofile("devil_angel.mp4",fps=20)
#TextClip(f""——•————————"",font ="edo",fontsize = 720//20,color = 'black').set_pos(('center',770)).set_start(counter).set_duration(angel_ly[i+1][0]-angel_ly[i][0])
#560
