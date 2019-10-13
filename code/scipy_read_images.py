import imageio
import glob, os
import numpy as np

import labelling
import config

def get_data():

	filelist=glob.glob(os.path.join(config.INPUT_DIR, "*.png"))
	sorted(filelist)
	features = {}
	labels=labelling.find_raga(config.INPUT_DIR)

	for i, filename in enumerate(filelist):
		features[i]=imageio.imread(filename)

	return features, labels
