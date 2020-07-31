from __future__ import print_function

import argparse
import glob
import os
import random

import cv2
import numpy as np
from pyimagesearch.rgbhistogram import RGBHistogram
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-f", "--females", required=True, help="number of images of females.(Max 5407)"
)
ap.add_argument(
    "-m", "--males", required=True, help="number of images of males.(Max 4372)"
)
args = vars(ap.parse_args())

path = "../../data/images"
imagePaths = sorted(glob.glob(path + "/*.jpg"))

# Sort out the data based on gender
# number of images of males: 4372
# number of images of females: 5407
males = []
females = []
for imagePath in imagePaths:
    if imagePath.split("_")[1] == "0":
        males.append(imagePath)
    if imagePath.split("_")[1] == "1":
        females.append(imagePath)

# Select random images from the sorted data. Number will be taken from the user.
random_females = random.sample(females, int(args["females"]))
random_males = random.sample(males, int(args["males"]))


imagePaths = np.concatenate((random_males, random_females))

desc = RGBHistogram([8, 8, 8])

data = []
target = []
for imagePath in imagePaths:
    # load the image
    image = cv2.imread(imagePath)
    # describe the image

    features = desc.describe(image)

    data.append(features.flatten())
    target.append(imagePath.split("_")[1])


# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)


# construct the training and testing splits
(trainData, testData, trainTarget, testTarget) = train_test_split(
    data, target, test_size=0.3, random_state=42
)

# train the classifier
model = SVC()
model.fit(trainData, trainTarget)

print("0: Male")
print("1: Female")
# evaluate the classifier
print(
    classification_report(testTarget, model.predict(testData), target_names=targetNames)
)

# loop over a sample of the images
for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
    # grab the image
    imagePath = imagePaths[i]

    # load the image
    image = cv2.imread(imagePath)

    # describe the image
    # features = desc.describe(image)
    features = desc.describe(image)

    gender = le.inverse_transform(model.predict([features.flatten()]))[0]
    print(imagePath)
    if gender == "0":
        print("The person on this photo might be male")
    else:
        print("The person on this photo might be female")
    cv2.imshow("image", image)
    cv2.waitKey(0)
