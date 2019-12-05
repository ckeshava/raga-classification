from threading import Thread
import librosa.display
import numpy as np
import json
from scipy import signal
import matplotlib.pyplot as plt
import os, glob
from multiprocessing import Pool
from datetime import datetime

AUDIO_FILES = "../carnatic"
JSON_FILES = AUDIO_FILES
CLIP_DURATION = 10
SKIP_DURATION = 10
SAMPLE_FREQUENCY = 44100

def get_all_ragas():
    # store all ragas in a dictionary
    json_files = glob.glob(os.path.join(JSON_FILES, "*.json"))
    assert len(json_files) > 0, "No json Files"

    all_ragas = {}

    for metadata_file in json_files:
        with open(metadata_file, 'r') as f:
            meta_info = json.load(f)

        if len(meta_info['raaga']) > 0:
            raga = meta_info['raaga'][0]['common_name']

            if raga not in all_ragas:
                all_ragas[raga] = 1
            else:
                all_ragas[raga] += 1

    return all_ragas


audio_files = glob.glob(os.path.join(AUDIO_FILES, "*.mp3"))
assert len(audio_files) > 0, "No Audio Files"

temp_ragas = get_all_ragas()
print(temp_ragas)
def get_count(elem):
    return temp_ragas[elem]

all_ragas = sorted(temp_ragas, key=get_count, reverse=True)

# consider only the top 20 ragas
RAGAS = all_ragas[:20]

for raga in RAGAS:
   print("{} : {}".format(raga, temp_ragas[raga]))
handle = open("completed_audio_files.txt", "a")




def prepare(wav_file, all_ragas=all_ragas, sr=44100):
    metadata_file = wav_file.replace('.mp3', '.json')
    with open(metadata_file, 'r') as f:
        meta_info = json.load(f)

    try:
        raga = meta_info['raaga'][0]['common_name']
    except IndexError:
        print("Uh Oh, no raga info. moving on!!")
        return
	

    if raga not in RAGAS:
        return
    # generate 3 seconds worth of audio from y
    clips = []

    y, sr = librosa.load(f, sr=44100)
    y = librosa.to_mono(y)
    y = librosa.util.normalize(y)
    # clipping with segments having 0.5 seconds difference in time
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
      
        if not os.path.exists(raga):
            os.makedirs(raga)
	# filename is a bithc
        plt.savefig(os.path.join(raga, str(datetime.now()) + ".png"))
        print("Finished {}/{} files".format(i, len(clips)))
    handle.write(wav_file)
    handle.write("\n")


def thread_handling(thread_id, audio_files=audio_files):
    i = thread_id

    while i < len(audio_files):
        print("processing files index: {}".format(i))
        f = audio_files[thread_id]
        prepare(f, all_ragas, SAMPLE_FREQUENCY)
        i += 10

p = Pool(5)
p.map(thread_handling, [0, 1, 2, 3, 4])
