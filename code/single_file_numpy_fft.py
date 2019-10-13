import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
import os, glob

import config

input_dir = config.BEGADA
filename = "/home/chenna/data/audio/varali/08-RTP-Varali.mp3"
x, sr = librosa.load(filename, sr=None)
sp = np.fft.fft(x)
freq = np.fft.fftfreq(x.shape[-1])
plt.plot(freq, sp.real, freq, sp.imag)
plt.savefig(filename[:-4] + ".png", bbox_inches='tight')

print(filename)

