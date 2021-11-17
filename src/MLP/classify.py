from __future__ import print_function

import argparse
import glob
import os
import random

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from pyimagesearch.rgbhistogram import RGBHistogram

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

    # features = desc.describe(image)

    mean = (104, 117, 123)
    features = cv2.dnn.blobFromImage(image, 1.0, (227, 227), mean, swapRB=False)

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
model = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100),
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="constant",
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=200,
    shuffle=True,
    random_state=None,
    tol=1e-4,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    max_fun=15000,
)
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

    features = cv2.dnn.blobFromImage(image, 1.0, (227, 227), mean, swapRB=False)

    gender = le.inverse_transform(model.predict([features.flatten()]))[0]
    print(imagePath)
    if gender == 0:
        print("The person on this photo might be male")
    else:
        print("The person on this photo might be female")
    cv2.imshow("image", image)
    cv2.waitKey(0)
