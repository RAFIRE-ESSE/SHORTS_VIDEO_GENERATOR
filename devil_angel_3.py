from flask import Flask, render_template, request,send_file
import cv2 
from moviepy.editor import *
from moviepy.config import change_settings
from mutagen.mp3 import MP3
import numpy as np
import os,numpy
from PIL import Image,ImageFilter
from os.path import isfile, join
import audio2numpy as a2n
from scipy.io.wavfile import write

devil=Flask(__name__)
devil.jinja_env.filters['zip'] = zip

def lyrics(file_devil):
    data=open(file_devil,'rb').read().decode('utf-8')
    data=[i.split(']') for i in data.split('\n') if i!=""]
    data=[[i[0].split('[')[1].split(':'),i[1]] for i in data]
    return [[int(z[0][0])*60+float(z[0][1]),z[1]] for z in data if z[0][0].isdigit()]

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

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
 
def of_devil(devil_audio,devil_ly,devil_image=None,font="#ffffff",back="#ffffff"):
    #Image.fromarray(numpy.array(hex_to_rgb(back)*(1080*720), dtype=numpy.uint8).reshape((1080,720,3)), 'RGB').save("angel.png")
    cv2.imwrite('devil_image.jpg',cv2.resize(cv2.imread(devil_image),(720,1080)))
    Image.open('devil_image.jpg').filter(ImageFilter.GaussianBlur(150)).save("angel.png")
    convert_frames_to_video(2)
    print(hex_to_rgb(back))
    x,devil_count,y,devil_,angel_ly,counter=80,0,119,[],lyrics(devil_ly),0
    clip,devil=VideoFileClip("video.mp4"),rect_with_rounded_corners(cv2.resize(cv2.cvtColor(cv2.imread(devil_image),cv2.COLOR_BGR2RGB),(500,500)),10,1,hex_to_rgb(back)) 
    devil_up=TextClip(f"——————————",font="Arial Black",fontsize = 720//15,color=font).set_pos(('center',700))
    devil_up_2=TextClip(f"||",font="Arial Black",fontsize = 720//18,color=font).set_pos(('center',630))
    for i in range(int(angel_ly[len(angel_ly)-1][0]-angel_ly[0][0])):
        devil_up_1=TextClip(f"•",fontsize = 720//19,color=font).set_pos((y,711))
        if i%2==0:
            for i in range(10):
                image=ImageClip(devil).set_position(("center",x))
                CompositeVideoClip([clip,image,devil_up,devil_up_1,devil_up_2]).save_frame(f"frame_devil/{devil_count}.png",t=1)
                x+=1
                devil_count+=1
        else:
            for i in range(10):
                image=ImageClip(devil).set_position(("center",x))
                CompositeVideoClip([clip,image,devil_up,devil_up_1,devil_up_2]).save_frame(f"frame_devil/{devil_count}.png",t=1)
                x-=1
                devil_count+=1
        print(devil_count)
        y+=(479)/(int(angel_ly[len(angel_ly)-1][0]-angel_ly[0][0]))
        image.close()

    video_=concatenate_videoclips([ImageClip(f"frame_devil/{i}.png").set_duration(0.10) for i in range(devil_count)])
    print(angel_ly)
    for i in range(len(angel_ly)):
        if i+1<len(angel_ly):
            if len(angel_ly[i][1])<=35:
                str_=angel_ly[i][1]
            else:
                devil=angel_ly[i][1].split(" ")
                str_,len_="",0
                for z in devil:
                    if len_<25:
                        str_+=f" {z}"
                        len_+=len(z)
                    else:
                        str_+=f"\n{z}"
                        len_=0
            devil_.append(TextClip(f"{str_}".title(),font ="Freshy Regular",fontsize = 720//20,color=font).set_pos(('center',800)).set_start(counter).set_duration(angel_ly[i+1][0]-angel_ly[i][0]))
            counter+=angel_ly[i+1][0]-angel_ly[i][0]
            print(counter)
            
    demon=CompositeVideoClip([clip]).set_audio(AudioFileClip(devil_audio)).subclip(0,angel_ly[len(angel_ly)-1][0]+1)
    demon.write_videofile("devil_angel.mp4",fps=20)
    devil__=VideoFileClip("devil_angel.mp4").subclip(angel_ly[0][0],angel_ly[len(angel_ly)-1][0])
    devil=CompositeVideoClip([video_]+devil_).set_audio(devil__.audio)
    devil.write_videofile("devil_angel.mp4",fps=20)
    video_.close(),devil__.close(),devil.close(),clip.close(),devil_up.close(),devil_up_1.close(),devil_up_2.close(),demon.close()
    del x,devil_count,y,devil_,angel_ly,counter,devil,devil__,devil_up_1,devil_up_2,video_,devil_up,demon
    

    
@devil.route("/div",methods=['POST','GET'])
def div():
    global f_color,b_color
    if request.method == "POST":
        index_out = request.form.getlist("devil")
        print(index_out)
        devil=open(f"devil/lyrics/{os.listdir('devil/lyrics')[0]}",'rb').read().decode('utf-8').split('\n')[int(index_out[0]):int(index_out[1])+2]
        out_order='\n'.join(devil)
        open(f"{os.listdir('devil/lyrics')[0]}",'wb').write(out_order.encode('utf-8'))
        out_order=lyrics(f"{os.listdir('devil/lyrics')[0]}")
        print(out_order[0],out_order[len(out_order)-1])
        if f_color!="" and b_color!="":
            of_devil(f"devil/song/{os.listdir('devil/song')[0]}",f"{os.listdir('devil/lyrics')[0]}",f"devil/image/{os.listdir('devil/image')[0]}",f_color,b_color)
        of_devil(f"devil/song/{os.listdir('devil/song')[0]}",f"{os.listdir('devil/lyrics')[0]}",f"devil/image/{os.listdir('devil/image')[0]}")
        return render_template("devil.html")

@devil.route("/reach_out",methods=['POST','GET'])
def reach_out():
    global f_color,b_color
    if request.method == 'POST' and request.files['ly'].filename!="":
        if request.files['song'].filename!="":
            for i in ['song','lyrics','image']:
                [os.remove(f"devil/{i}/{j}") for j in os.listdir(f'devil/{i}')]
            
            request.files['song'].save(f"devil/song/{request.files['song'].filename}")
            request.files['ly'].save(f"devil/lyrics/{request.files['ly'].filename}")
            request.files['image'].save(f"devil/image/{request.files['image'].filename}")
            f_color=request.form.get('font')
            b_color=request.form.get('back')
            data=open(f"devil/lyrics/{request.files['ly'].filename}",'rb').read().decode('utf-8')
            data=[f"[00:00.00]{f_color}"]+[']'.join(i) for i in [i.split(']') for i in data.split('\n') if i.split(']')[1]!=""]]
            open(f"devil/lyrics/{request.files['ly'].filename}",'wb').write('\n'.join(data).encode('utf-8'))
            print(data)
            print(f_color,b_color)
            return render_template("angel.html",devil=open(f"devil/lyrics/{request.files['ly'].filename}",'rb').read().decode('utf-8'))
        return render_template("devil.html")
    
        
@devil.route("/",methods=['POST','GET'])
def main_div():
    return render_template("devil.html")


if __name__ == '__main__':
    devil.run(debug=True)
    
#lyrics(f"devil/lyrics/{request.files['ly'].filename}",int(MP3(f"devil/song/{request.files['song'].filename}").info.length))
#of_devil(f"devil/{request.files['song'].filename}",f"devil/{request.files['ly'].filename}")
