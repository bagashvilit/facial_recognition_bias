# USAGE
# ** Locally Installed OpenCV: from `src` directory:
# python FaceRecognition.py --face cascades/haarcascade_frontalface_default.xml --image images/filename
# ** Docker Container: from a4_face-detection:
# docker run --rm -v $(pwd)/src:/root opencv python FaceRecognition.py --face cascades/haarcascade_frontalface_default.xml --image images/bts.jpg


# import the necessary packages
'''FaceDetector class is first defined. Then, we use argparse to parse
the command line arguments and cv2 to provide us with our OpenCV bindings.'''

from __future__ import print_function
from imagesearch.facedetector import FaceDetector
import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-i", "--image", required=True,
                help="path to where the image file resides")
args = vars(ap.parse_args())


'''Before we can even think about finding faces in an image,
we first need to define a class to handle how we are going
to find faces in an image. Go to facedetector!'''

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find faces in the image
'''instantiate the FaceDetector class,
supplying the path to our XML classifier as the sole parameter.'''
fd = FaceDetector(args["face"])
# detects the actual faces in the image by making a call to the detect method
faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=1,
                      minSize=(30, 30))
#  prints out the number of faces found in the image
print("I found {} face(s)".format(len(faceRects)))

# loop over the faces and draw a rectangle around eachpython FaceRecognition.py --face cascades/haarcascade_frontalface_default.xml --image images/filename
'''We need to loop over each individual face in order to
actually draw a bounding box around the image.
Each bounding box is just a tuple with four values:
the x and y starting location of the face in the image,
followed by the width and height of the face.'''
for (x, y, w, h) in faceRects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the detected faces
cv2.imwrite("images/output/Faces.png", image)
cv2.waitKey(0)
