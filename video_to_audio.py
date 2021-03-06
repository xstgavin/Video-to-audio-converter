#!/usr/bin/env python3
import os 
import matplotlib

import urllib.request
import urllib.error
import re
import sys
import time
import pipes
import pylab
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np
import glob
import subprocess
import json
import cv2
import torchaudio
import random

mel_option = 'v2'

def ffprobe(filename):
    command = ["ffprobe", "-v", "error", "-show_streams", "-print_format", "json", filename]
    xml = subprocess.check_output(command)
    jdat=json.loads(xml)
    return jdat



def audio_downsample_test(audio_file,delete = True,dsize=8):
    #reduce sample rate of the audio
    afname = audio_file.split('.wav')[0]
    jdat=ffprobe(audio_file)
    augments = []
    ars=['8000','11025','16000','22050','24000','32000','44100','48000']
    for ar in ars[:dsize]:
        dstName = afname + "_"+ar+"_aug.wav"
#        downsample = "ffmpeg  -i "+audio_file+" -b:a "+ba+" -acodec mp3 -ac 1 -ar "+\
#                ar+" "+dstName
        downsample = "ffmpeg  -v 0 -i "+audio_file+" -ac 1 -ar "+\
                ar+" "+dstName
        os.system(downsample) 
        melgram_v1(dstName,dstName[:-3]+'png')
        if(delete):
            os.system("rm "+dstName)

def audio_downsample(audio_file):
    #reduce sample rate of the audio
    afname = audio_file.split('.wav')[0]
    jdat=ffprobe(audio_file)
    augments = []
    if mel_option == 'v1':
        saveRoot = os.path.dirname(audio_file)+"/mel/"
    else:
        saveRoot = os.path.dirname(audio_file)+"/mel2/"
    if not os.path.exists(saveRoot):
        os.mkdir(saveRoot)
    ars=['8000','11025','16000','22050','24000','32000','44100','48000']
    for ar in ars:
        dstName = saveRoot+afname.split('/')[-1] + "_"+ar+".wav"
#        downsample = "ffmpeg  -i "+audio_file+" -b:a "+ba+" -acodec mp3 -ac 1 -ar "+\
#                ar+" "+dstName
        downsample = "ffmpeg  -v 0 -i "+audio_file+" -ac 1 -ar "+\
                ar+" "+dstName
        os.system(downsample) 
        if mel_option == 'v1':
            melgram_v1(dstName,dstName.split('.')[0]+'.png')
            os.system("rm "+dstName)
        else:
            print(dstName.split('.')[0]+'.png')
            melgram_v2(dstName,dstName.split('.')[0]+'.png')
            os.system("rm "+dstName)
           
 
def melgram_v2(audio_file_path, to_file):
    sig, fs = librosa.load(audio_file_path)
    a = random.randint(-6,6)
    sig = librosa.effects.pitch_shift(sig, fs, n_steps=a)
    #sig = sig+np.random.randn(len(sig))*0.001
    S = librosa.feature.melspectrogram(y=sig, sr=fs, n_mels=512,hop_length=512) # change hop_length to change y_axis
    
    D = (librosa.power_to_db(S, ref=np.max)+80)*2
    imgDat = np.flip(np.abs(np.round(D)),0).astype(np.uint8)
    #imgDat = cv2.resize(imgDat,(640,480))
    cvimg = cv2.imwrite(to_file,imgDat)

def melgram_v1(audio_file_path, to_file):
    sig, fs = librosa.load(audio_file_path)
    pylab.axis('on')  # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    D = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(D)
    if os.path.exists(to_file):
        os.remove(to_file)
    pylab.savefig(to_file, bbox_inches=None, pad_inches=0)
    pylab.close()
    melgram_v2(audio_file_path, to_file+'.png')

def video_to_audio(fileName):
    try:
        file, file_extension = os.path.splitext(fileName)
        file = pipes.quote(file)
        video_to_wav = 'ffmpeg -v 0 -i ' + file + file_extension + ' ' + file + '.wav'
        #final_audio = 'lame '+ file + '.wav' + ' ' + file + '.mp3'
        rm_wav = 'rm '+file+'.wav'
        #print(video_to_wav, final_audio)
        os.system(video_to_wav)
        #os.system(final_audio)
        #os.system(rm_wav)
        if os.path.exists(file+'.wav'):
            melgram_v1(file+'.wav',file+'.png')
        #file=pipes.quote(file)
        #os.remove(file + '.wav')
        print("sucessfully converted ", fileName, " into audio!")
    except OSError as err:
        print(err.reason)
        exit(1)

def extract_train_mel(pid=0):
    path = '/home/xiaoshengtao/hdd/DATA/deepfake-detection-challenge/train_videos/dfdc_train_part_'
#    for i in range(45,50):
    fPath = path + '%d/'%pid
    mp4s = glob.glob(fPath+'*.mp4')
    for mp4 in mp4s:
        video_to_audio(mp4)

def aug_org_mel(partid=1):
    jdat = json.load(open('./labels/all_label.json','r'))
    path = '/home/xiaoshengtao/hdd/DATA/deepfake-detection-challenge/train_videos/dfdc_train_part_'
    for key in jdat.keys():
        keyx = list(jdat[key].keys())
        ID=int(jdat[key][keyx[0]].split('/')[-1].split('_')[-1])
        if ID != partid:
            continue 
        fpath = path+"%d"%ID+"/"
        dstDir = fpath + "mel/"
        print(dstDir)
        if not os.path.exists(dstDir):
            os.mkdir(dstDir)
        mp4 = fpath+key
        wav = mp4[:-3]+'wav'
        if not os.path.exists(wav):      
            video_to_audio(mp4)
        if not os.path.exists(wav):
           continue
        audio_downsample(wav)

def aug_flac_mel(save_root= '/home/xiaoshengtao/audio/'):
    path = '/home/xiaoshengtao/hdd/DATA/audio/LibriSpeech/train-clean-100/'
    jdat = json.load(open(path+"flac.json",'r'))
    ars=['8000','11025','16000']

    for key in jdat.keys():
        durations = jdat[key]['durations']
        idx = np.argsort(durations)
        sample_count= 0
        flac_paths = []
        for i in range(-1,len(idx)-1):
            if sample_count>=6:
                break
            absIdx=idx[i]
            flac_path = path+jdat[key]['names'][absIdx][2:]
            if not os.path.exists(flac_path):
                print(flac_path+'missing')
            if durations[absIdx]<10:
                continue
            else:
                flac_paths.append(flac_path)
                sample_count = sample_count+1
        for flacPath in flac_paths:
            afname = flacPath.split('/')[-1]
            print(flacPath, afname)
            for ar in ars:
                dstName = save_root+afname.split('.')[0]+"_"+ar+".wav"
                downsample = "ffmpeg  -v 0 -i "+flacPath+" -ac 1 -ar "+ar+" "+dstName
                os.system(downsample) 
                melgram_v2(dstName,dstName.split('.')[0]+'.png')
                os.system("rm "+dstName)
      
def aug_train_mel():
    path = '/home/xiaoshengtao/hdd/DATA/deepfake-detection-challenge/train_videos/dfdc_train_part_'
    if len(sys.argv)==2:
        i  = int(sys.argv[1])
        fPath = path + '%d/'%i
        print(fPath)
        mp4s = glob.glob(fPath+'*.mp4')
        for mp4 in mp4s:
            wav = mp4[:-3]+"wav"
            if not os.path.exists(wav):
                print("video_to_audio "+mp4)
                video_to_audio(mp4)
            if not os.path.exists(wav):
                continue
            audio_downsample(wav)
    else:
        for i in range(45,50):
            fPath = path + '%d/'%i
            mp4s = glob.glob(fPath+'*.mp4')
            for mp4 in mp4s:
                wav = mp4[:-3]+"wav"
                if not os.path.exists(wav):
                    video_to_audio(mp4)
                audio_downsample(wav)

def main():
    if len(sys.argv) <1 or len(sys.argv) > 2:
        print('command usage: python3 video_to_audio.py FileName')
        exit(1)
    else:
        filePath = sys.argv[1]
        # check if the specified file exists or not
        try:
            if os.path.exists(filePath):
                print("file found!")
        except OSError as err:
            print(err.reason)
            exit(1)
        # convert video to audio
        if filePath.find('.wav') >=0 or filePath.find('.mp3')>=0 or filePath.find('.flac')>=0:
            print('just generate melgram')
            melgram_v1(filePath,filePath.split('.')[0]+'png')
            audio_downsample_test(filePath)        
        else:
            video_to_audio(filePath)
        time.sleep(1)
        
# install ffmpeg and/or lame if you get an error saying that the program is currently not installed 
if __name__ == '__main__':
    #extract_train_mel(int(sys.argv[1]))
    main()
    #aug_train_mel()
    #aug_flac_mel()
    #aug_org_mel(int(sys.argv[1]))
