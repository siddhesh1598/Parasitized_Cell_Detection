# import
import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
import config
from resnet import ResNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

# define hyperparameters
NUM_EPOCHS = 20
INIT_LR = 1e-1
BS = 32

def poly_decay(epoch):
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0

	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	return alpha

# calculate total #images in train, val and test datasets
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# initialize training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize validation and testing data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize resNet model and compile it
print("[INFO] initializing model...")
model = ResNet.build(64, 64, 3, 2, (3, 4, 6),
	(64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#  define callbacks
callbacks = [LearningRateScheduler(poly_decay)]

# train
print("[INFO] training model...")
H = model.fit(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS,
	callbacks=callbacks)

# evaluate 
print("[INFO] evaluating model...")
testGen.reset()
predIdxs = model.predict(testGen,
	steps=(totalTest // BS) + 1)

print(classification_report(
	testGen.classes,
	np.argmax(predIdxs, axis=1),
	target_names=testGen.class_indices.keys()))

# save model
model.save("model.h5")

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")