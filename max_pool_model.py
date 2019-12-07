from datetime import datetime
import pickle
from keras.utils import to_categorical
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
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model


RAGAS = ["kalyani", "pantuvarali", "kedaragaula", "thodi", "begada", "bhairavi", "mohana", "sankarabharana"]
NUM_RAGAS = len(RAGAS)

data = []
labels = []
ctr=0

for raga in RAGAS:
    inp_files = glob.glob(os.path.join(raga, "*.png"))
    temp = [ctr] * len(inp_files)
    ctr += 1
    print("Raga: {} \t Number of Input files: {}".format(raga, len(inp_files)))
    labels += temp

    for f in inp_files:
        img = load_img(f)  
        x = img_to_array(img)  
        data.append(x)
        # np.append(data, x, axis=-1)

print(len(labels))
print(len(data))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(480, 640, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))


    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(256, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    #model.add(Dense(32))
    #model.add(Activation('relu'))

#    model.add(Dense(32))
#    model.add(Activation('relu'))
#    model.add(Dropout(0.2))

    model.add(Dense(NUM_RAGAS))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # print(model.summary())
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


#logdir = "logs/scalars/" + str(datetime.now())
#tensorboard_callback = keras.callbacks.Tensorboard(logdir=logdir)
model = baseline_model()
print(model.summary())

#history=model.fit(np.array(X_train), np.array(y_train), validation_split=0.0, epochs=2, verbose=2, callbacks=[tensorboard_callback])
history=model.fit(np.array(X_train), np.array(y_train), shuffle=True, validation_split=0.20, epochs=200, verbose=2)
score = model.evaluate(np.array(X_test), np.array(y_test))
print(score)

model.save('fully_convolutional_model.h5')
#estimator = KerasClassifier(build_fn=baseline_model, epochs=1, batch_size=5, verbose=2)
#kfold = KFold(n_splits=10, shuffle=True)
#results = cross_val_score(estimator, np.array(data), np.array(labels), cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# save history to a file
with open('./trainHistoryDict_maxpool_dropout', 'wb') as file_pi:
    pickle.dump(history, file_pi)


#print(history.history.keys())
# summarize history for accuracy
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('trainvstest.png')
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('traintestloss.png')
