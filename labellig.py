import glob, os

ragas = ["sahana", "kalyani", "begada", "abhogi", "mohanam", "saveri", "sri"]

def find_raga(input_dir):
    spec = {}
    
    for filename in glob.glob(os.path.join(input_dir, "*.mp3")):
        for i in ragas:
            if filename.find(i) != -1:
                spec[filename] = i
                print("{} \t {}".format(filename, i))

    return spec

find_raga("/home/keshava/Desktop/work/ragas/carnatic_varnam_1.0/Audio")

    