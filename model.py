from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import os, glob
import numpy as np

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from keras.utils.vis_utils import plot_model


RAGAS = ["thodi", "kalyani", "begada", "bhairavi", "mohana", "sankarabharana"]
NUM_RAGAS = len(RAGAS)

data = []
labels = []


for raga in RAGAS:
    inp_files = glob.glob(os.path.join(raga, "*.png"))
    temp = [raga] * len(inp_files)
    print("Raga: {} \t Number of Input files: {}".format(raga, len(inp_files)))
    labels += temp

    for f in inp_files:
        img = load_img(f)  
        x = img_to_array(img)  
        data.append(x)
        # np.append(data, x, axis=-1)

# print(labels)
print(len(labels))
print(len(data))

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(480, 640, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # assume 3 ragas
    model.add(Dense(NUM_RAGAS))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # print(model.summary())
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

# model = baseline_model()

estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=5, verbose=2)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, np.array(data), np.array(labels), cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



