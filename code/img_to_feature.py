import sys
import json
import os, glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import imageio

import config


def get_feature_label(wav_file):
    metadata_file = wav_file.replace('.mp3', '.json')
    with open(metadata_file, 'r') as f:
        meta_info = json.load(f)
	
    img = wav_file.replace(".mp3", ".png")


    if os.path.exists(img):
        raga = meta_info['raaga'][0]['name']
    
        im = imageio.imread(img)
        
    print("File: {}\t Raga: {}".format(img, raga))
    return im, raga

def traverse_images():
    files = glob.glob(os.path.join(config.JSON_FILES, "*.png"))
    print(len(files)) 
    for f in files:
        feature, raga = get_feature_label(f)
	#print(type(feature))
	#print(feature.shape)
        print(raga)
        print("\n\n")

def raga_to_1_hot():
    pass


if __name__ == "__main__":
    #print("arguments: {}".format(sys.argv[1]))
    #write_image(sys.argv[1])
    traverse_images()
    print("entering")
