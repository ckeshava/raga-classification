import imageio
import glob, os
import numpy as np
import matplotlib.pyplot as plt


import config

def get_img_data(raga):
    """ 
    Returns the images of all the fft's of a raga
    Args:
    raga: config.IMG_<raga_name>

    Returns:
    nd array of features
    """
    filelist=glob.glob(os.path.join(raga, "*.png"))
    sorted(filelist)
    print(len(filelist))
    features = []

    for i, filename in enumerate(filelist):
        features.append(imageio.imread(filename))
	
    return features

def test_get_img_data():
    begada_fft = get_img_data(config.IMG_BEGADA)

    print(len(begada_fft))
    for i in range(len(begada_fft)):
        print(begada_fft[i].shape)

#test_get_img_data()
