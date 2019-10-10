import glob, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

import matplotlib_spec 
import config

# Call this function from within the graph. TF does not accept tensors into placeholders
def conv_to_one_hot(indices):
    
    """ 
    Args:
    indices: a list of indices from [0, NUM_RAGAS)

    Returns:
    2 dimensional matrix with indices converted into 1-hot vectors
    """

    return tf.one_hot(indices=indices, depth=config.NUM_RAGAS)

def get_image(input_dir):
    return matplotlib_spec.get_spectrogram(input_dir)

def get_label(input_dir):
    """ returns a dict of {filename: raga_name} """
    
    label = []

    for filename in glob.glob(os.path.join(input_dir, "*.png")):

        # TO CHANGE:
        # label[filename] = np.random.randint(low=0, high=config.NUM_RAGAS)
        label.append(np.random.randint(low=0, high=config.NUM_RAGAS))

    return label

# get_image("input/")
