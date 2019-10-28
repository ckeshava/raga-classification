import sys
import json
import os, glob
import librosa
import numpy as np
import matplotlib.pyplot as plt


import config


def write_image(wav_file):
	#wav_file = os.path.join(config.JSON_FILES, wav_file)
	# get the raga name from the corresponding json file
	metadata_file = wav_file.replace('.mp3', '.json')
	with open(metadata_file, 'r') as f:
		meta_info = json.load(f)
		
	if len(meta_info['raaga']) > 0:
		raga = meta_info['raaga'][0]['name']

		# extract the fft from the mp3 file
		x, sr = librosa.load(wav_file, sr=None)
		sp = np.fft.fft(x)
		freq = np.fft.fftfreq(x.shape[-1])
		plt.plot(freq, sp.real, freq, sp.imag)
		img = wav_file.replace(".mp3", ".png")
		if not os.path.isdir(raga):
			os.makedirs(raga)	
			
		plt.savefig(img)
		print("File: {}\t Raga: {}".format(wav_file, raga))

if __name__ == "__main__":
	print("arguments: {}".format(sys.argv[1]))
	write_image(sys.argv[1])
