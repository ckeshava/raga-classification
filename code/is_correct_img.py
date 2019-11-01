import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import matplotlib.image as mpimg
import sys
import json
import os, glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

import config

counter = 0
ragas = {}

def get_feature_label(img_file):
    metadata_file = img_file.replace('.png', '.json')
    with open(metadata_file, 'r') as f:
        meta_info = json.load(f)
	


    if os.path.exists(img_file):
        raga = meta_info['raaga'][0]['name']
    
#         im = imageio.imread(img_file)
        # im = mpimg.imread(img_file)
        im = cv2.imread(img_file)
        print(type(im))        
        print("File: {}\t Raga: {}".format(img_file, raga))
        print(im)
        #cv2.imshow('image', im)         
        #cv2.waitKey(0)         
        #cv2.destroyAllWindows()  

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

traverse_images()
