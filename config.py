# UNITS must be the product of the length and breadth of the spectrogram image 
LENGTH = 448
BREADTH = 1634
NUM_CHANNELS = 4
UNITS = LENGTH * BREADTH * NUM_CHANNELS
HIDDEN_UNITS = 2 * 64
NUM_RAGAS = 40
# Number of input samples. This could be thought of as the batch size
INPUT_SIZE = 5

INPUT_DIR = "/home/keshava/Desktop/work/ragas/carnatic_varnam_1.0/Audio/"

EPOCHS = 5 # Number of iterations per data sample
RAGAS = ["sahana", "kalyani", "begada", "abhogi", "mohanam", "saveri", "sri"]

MP3_FILES = "/home/student/Desktop/iTuned - Carnatic Instrumental"

