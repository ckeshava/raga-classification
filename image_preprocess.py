import glob, os
import matplotlib.pyplot as plt

import matplotlib_spec 



def get_image(input_dir):
    return matplotlib_spec.get_spectrogram(input_dir)

def get_label():
    """ returns a dict of {filename: raga_name} """
    pass

# get_image("input/")
