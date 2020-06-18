# import
import os

# initialize path to the original dataset
ORIG_INPUT_DATASET = "cell_images"

# initialize path to new dataset
BASE_PATH = "maleria"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the train and validation split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1