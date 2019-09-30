# https://www.kaggle.com/ashishpatel26/feature-extraction-from-audio
import librosa
import sklearn
import matplotlib.pyplot as plt
import librosa.display
import glob, os

import warnings
warnings.filterwarnings('ignore')

import config

# Todo: change to relative path 
# filename="/home/keshava/Desktop/work/ragas/carnatic_varnam_1.0/Audio/223578__gopalkoduri__carnatic-varnam-by-dharini-in-abhogi-raaga.mp3"

filelist = glob.glob(os.path.join(config.CARNATIC_INPUT_DIR, "*.mp3"))
print(len(filelist))
for i, filename in enumerate(filelist, start=0):
    print(i)
    print(filename)
    x, sr = librosa.load(filename, sr=None)

    # # spectrogram
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    # plt.figure(figsize=(20, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    # # plt.colorbar()

    # plt.savefig(str(i) + ".png", bbox_inches='tight')


