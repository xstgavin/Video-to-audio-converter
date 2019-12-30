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


def melgram_v1(audio_file_path, to_file):
	sig, fs = librosa.load(audio_file_path)
	pylab.axis('on')  # no axis
	pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
	S = librosa.feature.melspectrogram(y=sig, sr=fs)
	librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
	pylab.savefig(to_file, bbox_inches=None, pad_inches=0)
	pylab.close()


def video_to_audio(fileName):
	try:
		file, file_extension = os.path.splitext(fileName)
		file = pipes.quote(file)
		video_to_wav = 'ffmpeg -i ' + file + file_extension + ' ' + file + '.wav'
		#final_audio = 'lame '+ file + '.wav' + ' ' + file + '.mp3'
		rm_wav = 'rm '+file+'.wav'
		#print(video_to_wav, final_audio)
		os.system(video_to_wav)
		#os.system(final_audio)
		os.system(rm_wav)
		if os.path.exists(file+'.wav'):
			melgram_v1(file+'.wav',file+'.png')
		#file=pipes.quote(file)
		#os.remove(file + '.wav')
		print("sucessfully converted ", fileName, " into audio!")
	except OSError as err:
		print(err.reason)
		exit(1)

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
		if filePath.find('.wav') >=0 or filePath.find('.mp3'):
			print('just generate melgram')
			melgram_v1(filePath,filePath[:-3]+'png')
		else:
			video_to_audio(filePath)
		time.sleep(1)
		
# install ffmpeg and/or lame if you get an error saying that the program is currently not installed 
if __name__ == '__main__':
	main()
