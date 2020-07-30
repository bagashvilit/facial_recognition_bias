# USAGE
# python classify.py --females number --males number
# number here should be desired number of images for each binary gender group
from __future__ import print_function
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2
import random

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--females", required = True,
	help = "number of images of females.(Max 5407)")
ap.add_argument("-m", "--males", required = True,
	help = "number of images of males.(Max 4372)")
args = vars(ap.parse_args())

path = "../../data/images"
imagePaths = sorted(glob.glob(path + "/*.jpg"))

males = []
females = []
for imagePath in imagePaths:
    if (imagePath.split("_")[1] == "0"):
        males.append(imagePath)
    if (imagePath.split("_")[1] == "1"):
        females.append(imagePath)

random_females = random.sample(females, int(args["females"]))
random_males = random.sample(males, int(args["males"]))

imagePaths = np.concatenate((random_males, random_females))
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')


target = []
labels = []
for imagePath in imagePaths:
    # load the image
    image = cv2.imread(imagePath)
    # describe the image

    mean = (104, 117, 123)
    blob = cv2.dnn.blobFromImage(image, 1.0, (227,227), mean, swapRB=False)

    target.append(imagePath.split("_")[1])

    gender_list = [0, 1]

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    labels.append(gender)

# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# evaluate the classifier
print(classification_report(target, labels,
	target_names = targetNames))
