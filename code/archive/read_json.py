import json
import os, glob
import librosa
import numpy as np
import matplotlib.pyplot as plt


import config


def write_image():
	wav_files=glob.glob(os.path.join(config.JSON_FILES, "*.mp3"))

	for i, wav_file in enumerate(wav_files):

		# get the raga name from the corresponding json file
		metadata_file = wav_file.replace('.mp3', '.json')
		with open(metadata_file, 'r') as f:
			meta_info = json.load(f)
		
		print("{}/{}\tFile: {}\n".format(i, len(wav_files), wav_file))
		if len(meta_info['raaga']) > 0:
			raga = meta_info['raaga'][0]['name']
			print("File: {}\t Raga: {}\n\n".format(wav_file, raga))

			# extract the fft from the mp3 file
			x, sr = librosa.load(wav_file, sr=None)
			sp = np.fft.fft(x)
			freq = np.fft.fftfreq(x.shape[-1])
			plt.plot(freq, sp.real, freq, sp.imag)
			img = wav_file.replace(".mp3", ".png")
			if not os.path.isdir(raga):
				os.makedirs(raga)	
			
			plt.savefig(img)


write_image()
