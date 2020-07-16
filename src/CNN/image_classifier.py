"""
Classify the provided image based on gender.

To run the program you need to have OpenCV installed
navigate to src/CNN and run the following command:
python image_classifier.py --image images/filename.jpg
"""

import Convolution_neural_net
import argparse
import cv2

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=False,
                help="path to the image")

args = vars(ap.parse_args())
image = cv2.imread(args["image"])
# please refer to Convolution_neural_net.py to see how classifiaction happens
print("Gender: ", Convolution_neural_net.gender_detector(image))
