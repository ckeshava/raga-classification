import librosa.display
import numpy as np
import json
from scipy import signal
import matplotlib.pyplot as plt
import os, glob
from datetime import datetime

import config

def get_all_ragas():
    # store all ragas in a dictionary
    json_files = glob.glob(os.path.join(config.JSON_FILES, "*.json"))
    assert len(json_files) > 0, "No json Files"

    counter = 0
    all_ragas = {}

    for metadata_file in json_files:
        with open(metadata_file, 'r') as f:
            meta_info = json.load(f)

        if len(meta_info['raaga']) > 0:
            raga = meta_info['raaga'][0]['name']

            if raga not in all_ragas:
                all_ragas[raga] = counter
                counter += 1

    return all_ragas



def prepare(y, wav_file, all_ragas, sr=44100):
    y = librosa.to_mono(y)
    y = librosa.util.normalize(y)

    # generate 3 seconds worth of audio from y
    clips = []

    # clipping with segments having 0.5 seconds difference in time
    for i in range(0, len(y) - (config.CLIP_DURATION - 1) * sr, config.SKIP_DURATION * sr):
        clips.append(y[i: i + config.CLIP_DURATION * sr])

    # find the json file for correct naming
    metadata_file = wav_file.replace('.mp3', '.json')
    with open(metadata_file, 'r') as f:
        meta_info = json.load(f)


    if len(meta_info['raaga']) > 0:
        raga = meta_info['raaga'][0]['name']


        # perform spectrogram on each of the segments

        for i, y in enumerate(clips):        
            # discard smaller segments (the end parts)
            if len(y) < config.CLIP_DURATION * sr:
                continue

            # plt.subplot(4, 2, 2)
            # librosa.display.specshow(y, y_axis='log')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('Log-frequency power spectrogram')
            # plt.savefig(filename.replace(".mp3", ".png"))
            # plt.show()


            frequencies, times, spectrogram = signal.spectrogram(y, sr)
            plt.pcolormesh(times, frequencies, spectrogram)
            plt.imshow(spectrogram)
            plt.yscale('log')
            plt.xlim(0, config.CLIP_DURATION)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            # filename is a bithc
            if raga not in all_ragas:
                id = str(111)
            else:
                id = str(all_ragas[raga])
            plt.savefig(id + "_" + str(datetime.now()) + ".png")
            # plt.show()
            print("Finished {}/{} files".format(i, len(y)))




audio_files = glob.glob(os.path.join(config.AUDIO_FILES, "*.mp3"))
assert len(audio_files) > 0, "No Audio Files"

all_ragas = get_all_ragas()
print("DEBUG")
print(all_ragas)
for i, f in enumerate(audio_files):
    y, sr = librosa.load(f, sr=44100)
    prepare(y, f, all_ragas, config.SAMPLE_FREQUENCY)
    print("="*30)
    print("\n\nCompleted processing file: {}/{}\n\n".format(i, len(audio_files)))
    print("="*30)
