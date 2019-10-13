# https://www.kaggle.com/ashishpatel26/feature-extraction-from-audio
import librosa
import matplotlib.pyplot as plt
import librosa.display



# Todo: change to relative path 
filename="/home/keshava/Desktop/work/ragas/carnatic_varnam_1.0/Audio/223605__gopalkoduri__carnatic-varnam-by-vignesh-in-sri-raaga.mp3"

x, sr = librosa.load(filename, sr=None)

# # spectrogram
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(20, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')

plt.savefig(filename[:-4] + ".png", bbox_inches='tight')
print(filename)





