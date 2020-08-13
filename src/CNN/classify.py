import argparse

import cv2


def load_caffe_models():
    """
    This function loads caffe model.

    .prototxt file holds overall information about the neural network, such as:
    list of layers, connections between layers,parameters of each layers, and input/output dimensions.
    .caffemodel file stores weights of layers of Neural network
    """
    gender_net = cv2.dnn.readNetFromCaffe(
        "deploy_gender.prototxt", "gender_net.caffemodel"
    )
    return gender_net


def classify(imagePaths):
    target = []
    label = []
    for imagePath in imagePaths:
        gender_net = load_caffe_models()

        gender_list = ["0", "1"]
        """
        Preprocess image and prepare it for the classification.

        scalefactor: after mean subtraction image can be scaled by the scalefactor
        size: size that Convolutional Neural Network expects.
        mean values here are for mean subtraction, usually the mean values that were used for
        training are used for classification as well
        """

        image = cv2.imread(imagePath)

        mean = (104, 117, 123)
        blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), mean, swapRB=False)

        target.append(imagePath.split("_")[1])
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        label.append(gender)
    return (target, label)
