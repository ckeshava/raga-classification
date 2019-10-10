# Source: https://pythontic.com/visualization/signals/spectrogram
# Thumb Rule: High frequency resolution ===> larger window

# OPTIMIZATION HYPER PARAM:
# Understand the knobs used in the matplotlib spectrogram function. Optimisation required.

# run the get_mono_wav function exactly once on all of the input files.

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment
import os, glob

# might have to change this value according to dataset
sample_rate = 16000

def preprocess_input(input_dir):
    """Proprocess all stereo files into mono"""
    for filename in glob.glob(os.path.join(input_dir, "*.wav")):
        get_mono_wav(filename)


def get_mono_wav(filename):
    """ convert stereo to mono """
    sound = AudioSegment.from_wav(filename)
    sound = sound.set_channels(1)
    sound.export(os.path.join(filename), format="wav")

def get_spectrogram(input_dir):
    """ Produces spectrogram for the audio files within input_dir """
    preprocess_input(input_dir)
    spec = []

    for filename in glob.glob(os.path.join("input/", "*.wav")):
        _, samples = wavfile.read(filename)

        # defaults parameter values are used
        Pxx, freq, bins, im = plt.specgram(samples, Fs=sample_rate)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.savefig(filename[:-4] + ".png")

        spec.append(Pxx)

    return spec

# get_spectrogram("input/")







