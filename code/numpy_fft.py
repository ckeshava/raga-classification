import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
import os, glob

import config

input_dir = config.VARALI
print(input_dir)
#files = glob.glob(os.path.join(input_dir, "*.mp3"))
#files = glob.glob('**/*.mp3', recursive=True)
files = [f for f in glob.glob(input_dir + "**/*.mp3", recursive=True)]
print(len(files))

#for filename in glob.glob(os.path.join(input_dir, "*.mp3")):
#filename = "/home/chenna/data/05 - Thyagaraja Namaste.mp3"
#x, sr = librosa.load(filename, sr=None)
#sp = np.fft.fft(x)
#freq = np.fft.fftfreq(x.shape[-1])
#plt.plot(freq, sp.real, freq, sp.imag)
#plt.savefig("trial.png", bbox_inches='tight')

#print(filename)


