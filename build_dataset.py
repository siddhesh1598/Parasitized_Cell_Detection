# import
import random
from imutils import paths
import shutil
import os
import config

# grab the path to all input images
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# compute training and testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# compute validation dataset from the training dataset
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# define the datasets 
datasets = [
	("training", trainPaths, config.TRAIN_PATH),
	("validation", valPaths, config.VAL_PATH),
	("testing", testPaths, config.TEST_PATH),
]

# loop over the dataset
for (dType, imagePaths, baseOutput) in datasets:
	print("[INFO] building '{}' split".format(dType))

	# create the output directory if does not exists
	if not os.path.exists(baseOutput):
		print("[INFO] creating {} directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input images
	for imagePath in imagePaths:
		# get the filename and label
		fileName = imagePath.split(os.path.sep)[-1]
		label = imagePath.split(os.path.sep)[-2]

		# build a path to label directory
		labelPath = os.path.sep.join([baseOutput, label])

		# create labeldirectory is does not exists
		if not os.path.exists(labelPath):
			print("[INFO] creating {} directory".format(labelPath))
			os.makedirs(labelPath)

		# construct path to the image and copy the original image
		p = os.path.sep.join([labelPath, fileName])
		shutil.copy2(imagePath, p)