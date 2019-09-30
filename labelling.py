import glob, os
import numpy as np

ragas = ["sahana", "kalyani", "begada", "abhogi", "mohanam", "saveri", "sri"]

def find_raga(input_dir):
    filelist = glob.glob(os.path.join(input_dir, "*.png"))
    sorted(filelist)
    spec = np.zeros(len(filelist))
    for i, filename in enumerate(filelist):
        for j in range(len(ragas)):
            if filename.find(ragas[j]) != -1:
                spec[i] = int(j) # by default, it is stored as int !??

    return spec


    
