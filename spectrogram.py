# Spectrogram in Audacity: https://manual.audacityteam.org/man/spectrogram_view.html
# Thumb Rule: High frequency resolution ===> larger window


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
import os

# might have to change this value according to dataset
sample_rate = 16000

def get_mono_wav(filename):
    """ convert stereo to mono """
    sound = AudioSegment.from_wav(filename)
    sound = sound.set_channels(1)

    sound.export(os.path.join("mono/", filename), format="wav")
    return os.path.join("mono/", filename)

    


filename = get_mono_wav('file_example_WAV_10MG.wav')
_, samples = wavfile.read(filename)
template = ('DEBUG: {} shape: {}')
# print(template.format(samples, samples.shape))
print(samples.shape)

# set the window value and noverlap values. 
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, noverlap=0)
spectrogram = np.log(spectrogram)
print(template.format(spectrogram, spectrogram.shape))

plt.imshow(spectrogram)
plt.pcolormesh(times, frequencies, spectrogram)

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
