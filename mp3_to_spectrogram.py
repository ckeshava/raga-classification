# https://www.kaggle.com/ashishpatel26/feature-extraction-from-audio
import librosa
import matplotlib.pyplot as plt
import librosa.display
from pathlib import Path
import glob, os
import numpy as np

import config

files_dir = config.MP3_FILES
ragas = config.RAGAS

count = np.zeros((len(ragas)))


for filename_path in Path(files_dir).glob('**/*.mp3'):

    filename = str(filename_path)
    for j in range(len(ragas)):
            if filename.find(ragas[j]) != -1:
                count[j] += 1 # by default, it is stored as int !??


                try:
                    x, sr = librosa.load(filename, sr=None)
                    
                    # # spectrogram
                    X = librosa.stft(x)
                    Xdb = librosa.amplitude_to_db(abs(X))
                    plt.figure(figsize=(20, 5))
                    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')

                    plt.savefig(filename[:-4] + ".png", bbox_inches='tight')
                    print(filename)

                except RuntimeError:
                    print('Invalid Format')



for i in range(len(ragas)):
    print("{} : {}".format(ragas[i], count[i]))



# # Todo: change to relative path 
# for filename in glob.glob(files_dir, '*.mp3', recursive=True):
#     # filename="/home/keshava/Desktop/work/ragas/carnatic_varnam_1.0/Audio/223605__gopalkoduri__carnatic-varnam-by-vignesh-in-sri-raaga.mp3"

#     # x, sr = librosa.load(filename, sr=None)

#     # # # spectrogram
#     # X = librosa.stft(x)
#     # Xdb = librosa.amplitude_to_db(abs(X))
#     # plt.figure(figsize=(20, 5))
#     # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')

#     # plt.savefig(filename[:-4] + ".png", bbox_inches='tight')
#     print(filename)
