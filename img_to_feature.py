import tensorflow as tf
import sys
import json
import os, glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import imageio

import config

counter = 0
ragas = []


def get_feature_label(img_file):
    metadata_file = img_file.replace('.png', '.json')
    with open(metadata_file, 'r') as f:
        meta_info = json.load(f)
	


    if os.path.exists(img_file):
        raga = meta_info['raaga'][0]['name']
    
        im = imageio.imread(img_file)
        
        print("File: {}\t Raga: {}".format(img_file, raga))
        return im, raga

def traverse_images():
    files = glob.glob(os.path.join(config.JSON_FILES, "*.png"))
    print(len(files)) 
    ans = []
    for f in files:
        feature, raga = get_feature_label(f)
	

        print("File: {}\tRaga: {}".format(f, raga))
        ans.append((feature, raga))
    
    return ans


def raga_to_1_hot(data):
    # store all ragas in a dictionary
    k = 0
    ragas = {}
    for i in range(len(data)):
        if not data[i][1] in ragas:
            ragas[data[i][1]] = k
            k += 1 

    ans = []    
    for i in range(len(data)):
        temp = np.zeros(k)
        temp[ragas[data[i][1]]] = 1

        ans.append((data[i][0], temp))

    return ans
        
    


if __name__ == "__main__":
    ans = traverse_images()
    ans = raga_to_1_hot(ans)
    print("done!")
