import argparse
import glob
import random

import classify
import cv2
import numpy as np
from sklearn.metrics import classification_report

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

target, label = classify.classify(imagePaths)

targetNames = np.unique(target)

print(classification_report(target, label, target_names=targetNames))
