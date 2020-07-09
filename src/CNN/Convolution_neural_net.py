"""
Demonstrate gender classification using Convolution nerual network.

Framework: Caffe. model can also be loaded with OpenCV.
link to the source code: https://github.com/GilLevi/AgeGenderDeepLearning

"""
import cv2


def load_caffe_models():
    # .prototxt file holds overall information about the neural network, such as:
    # list of layers, connections between layers,parameters of each layers, and input/output dimensions.
    # .caffemodel file stores weights of layers of Neural network
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(gender_net)


def gender_detector(image):

    gender_net = load_caffe_models()

    gender_list = ['Male', 'Female']

    #Preprocess image and prepare it for the classification
    # scalefactor: after mean subtraction image can be scaled by the scalefactor
    # size: size that Convolutional Neural Network expects.
    # mean values here are for mean subtraction, usually the mean values that were used for
    # training are used for classification as well
    mean = (104, 117, 123)
    blob = cv2.dnn.blobFromImage(image, 1.0, (227,227), mean, swapRB=False)

    #Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    return gender

# provide the path to the input image as a parameter.
# The output will be Open CV window with a label on top, label is generated
# using the convolution neural network
image = cv2.imread("images/sample.jpg")
gender = gender_detector(image)
cv2.imshow("Gender: {}".format(gender),image)
cv2.waitKey()
