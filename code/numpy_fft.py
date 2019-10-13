import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
import os, glob

import config

input_dir = config.VARALI
files = glob.glob(os.path.join(input_dir, "*.mp3"))
k = 0
for filename in files:
    x, sr = librosa.load(filename, sr=None)
    sp = np.fft.fft(x)
    freq = np.fft.fftfreq(x.shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag)
    plt.savefig(str(k) + ".png", bbox_inches='tight')
    k += 1

    print(filename)

