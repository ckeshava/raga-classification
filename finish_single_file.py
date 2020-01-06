import librosa.display
import numpy as np
import json
from scipy import signal
import matplotlib.pyplot as plt
import os, glob
from datetime import datetime
import sys

AUDIO_FILES = "."
JSON_FILES = AUDIO_FILES
CLIP_DURATION = 10
SKIP_DURATION = 10
SAMPLE_FREQUENCY = 44100


def prepare(wav_file, sr=44100):
    # generate 3 seconds worth of audio from y
    clips = []

    try:
        y, sr = librosa.load(wav_file, sr=44100)
        y = librosa.to_mono(y)
        y = librosa.util.normalize(y)
        for i in range(0, len(y) - (CLIP_DURATION - 1) * sr, SKIP_DURATION * sr):
            clips.append(y[i: i + CLIP_DURATION * sr])


        for i, y in enumerate(clips):        
        # discard smaller segments (the end parts)
            if len(y) < CLIP_DURATION * sr:
                continue

            frequencies, times, spectrogram = signal.spectrogram(y, sr)
            plt.pcolormesh(times, frequencies, spectrogram)
            plt.imshow(spectrogram)
            plt.yscale('log')
            plt.xlim(0, CLIP_DURATION)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
      
            plt.savefig(str(datetime.now()) + ".png")
            print("Finished {}/{} files".format(i, len(clips)))

    except RuntimeError:
        pass

    # clipping with segments having 0.5 seconds difference in time
audio_files = glob.glob('*.mp3')

for f in audio_files:
    prepare(f)
    print("="*50)
    print("completed {}".format(f))
    print("="*50)
