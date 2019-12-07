from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os, glob
from keras.preprocessing.image import img_to_array, load_img

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

import pickle
f = open("X_train", "wb")
pickle.dump(X_train, f)
f.close()

f = open("y_train", "wb")
pickle.dump(y_train, f)
f.close()

f = open("X_test", "wb")
pickle.dump(X_test, f)
f.close()

f = open("y_test", "wb")
pickle.dump(y_test, f)
f.close()
