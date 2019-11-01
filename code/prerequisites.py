import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import matplotlib.image as mpimg
import sys
import json
import os, glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2

import config

import models

counter = 0
ragas = {}

def get_feature_label(img_file):
    metadata_file = img_file.replace('.png', '.json')
    with open(metadata_file, 'r') as f:
        meta_info = json.load(f)
	


    if os.path.exists(img_file):
        raga = meta_info['raaga'][0]['name']
    
#         im = imageio.imread(img_file)
        im = cv2.imread(img_file)
        
        #print("File: {}\t Raga: {}".format(img_file, raga))
        return im, raga


def traverse_images():
    files = glob.glob(os.path.join(config.JSON_FILES, "*.png"))
    ans = []
    for f in files:
        feature, raga = get_feature_label(f)
        ans.append((feature, raga))
    
    return ans


def raga_to_1_hot(data):
    # store all ragas in a dictionary
    counter = 0
    # all_ragas = []
    for i in data:
        if not i[1] in ragas:
            ragas[i[1]] = counter
            # all_ragas.append(i[1])
            counter += 1 

    # all_ragas = set(all_ragas)
    # print(ragas)

    # print(all_ragas)
    
    new_data = []
    for i in range(len(data)):
        temp = ragas[data[i][1]]
        new_data.append((data[i][0], [temp]))
    return np.array(new_data)

ans = traverse_images()        
ans = raga_to_1_hot(ans)

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

images = []
labels = []

for i in range(len(ans)):
    
    # required only for trees, forests
    # images.append(np.resize(ans[i][0], [ 480 * 640 * 4]))  
    images.append(ans[i][0])
    labels.append(ans[i][1])

images = np.array(images)
labels = np.array(labels)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)

# print("DEBUG: x_train shape: {}".format(x_train.shape))
# print("DEBUG: x_test shape: {}".format(x_test.shape))
# print("DEBUG: y_train shape: {}".format(y_train.shape))
# print("DEBUG: y_test shape: {}".format(y_test.shape))
# all of the above is common to all non-neural network models

# # Random Forest
# print("==" * 30)
# print("Random Forest")
# clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# clf.fit(x_train, y_train)

# print(clf.feature_importances_)
# score = clf.score(x_test, y_test)

# print(score)
print("==" * 30)

print("CNN Model")
# CNN Model
models.CNNModel(np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))
print("==" * 30)
