from __future__ import print_function

import argparse
import glob
import random

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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


# load caffemodel
# .prototxt file holds overall information about the neural network, such as:
# list of layers, connections between layers,parameters of each layers, and input/output dimensions.
# .caffemodel file stores weights of layers of Neural network
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")


target = []
labels = []
for imagePath in imagePaths:
    # load the image
    image = cv2.imread(imagePath)
    # describe the image
    # scalefactor: after mean subtraction image can be scaled by the scalefactor
    # size: size that Convolutional Neural Network expects.
    # mean values here are for mean subtraction, usually the mean values that were used for
    # training are used for classification as well
    mean = (104, 117, 123)
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), mean, swapRB=False)

    target.append(imagePath.split("_")[1])

    gender_list = [0, 1]

    # Predict the gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    labels.append(gender)

# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)


print("0: Male")
print("1: Female")
# evaluate the classifier
print(classification_report(target, labels, target_names=targetNames))


for i in np.random.choice(np.arange(0, len(imagePaths)), 10):
    # grab the image
    imagePath = imagePaths[i]

    # load the image
    image = cv2.imread(imagePath)

    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), mean, swapRB=False)

    gender_list = [0, 1]

    # Predict the gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    print(imagePath)
    if gender == 0:
        print("The person on this photo might be male")
    else:
        print("The person on this photo might be female")
    cv2.imshow("image", image)
    cv2.waitKey(0)
