#!/usr/bin/env python
# coding: utf-8

# In[7]:


# import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras import layers

import matplotlib.image as mpimg
import sys
import json
import os, glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import imageio

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
        im = mpimg.imread(img_file)
        
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
    counter = 0
    for i in data:
        if not i[1] in ragas:
            ragas[i[1]] = counter
            counter += 1 
    
    new_data = []
    for i in data:
        temp = tf.one_hot(indices=ragas[i[1]], depth=40)
        new_data.append((i[0], temp))
    return new_data

ans = traverse_images()        
# ans = raga_to_1_hot(ans)


def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

images = []
labels = []

for i in range(len(ans)):
    images.append(ans[i][0])    
    labels.append(ans[i][1])
    
# print(images)
# print(labels)


NUMERIC_COLUMNS = ['features']
CATEGORICAL_COLUMNS = ['ragas']
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = np.unique(np.array(labels))
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))


for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32, shape=(480, 640, 4)))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensors((data_df, label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


train_input_fn = make_input_fn(images, labels)


# In[4]:



ds = make_input_fn(np.array(images), np.array(labels), batch_size=10)()

# for feature_batch, label_batch in ds.take(1):
# #     print(feature_batch.shape)
# #     print(label_batch.shape)
    
#     print()
#     print('A batch of class:', feature_batch.numpy())
#     print()
#     print('A batch of Labels:', label_batch.numpy())
# #     tf.keras.layers.DenseFeatures([feature_columns[0]])(feature_batch).numpy()



linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)


# linear_est.train(train_input_fn)
# result = linear_est.evaluate(train_input_fn)
# 
# clear_output()
# print(result)

# In[12]:


# feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
feature_layer = tf.compat.v2.keras.layers.DenseFeatures(feature_columns)


# In[ ]:




