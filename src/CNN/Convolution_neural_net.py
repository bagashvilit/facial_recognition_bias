import cv2

def load_caffe_models():
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(gender_net)



def gender_detector(image):
    gender_net = load_caffe_models()

    gender_list = ['Male', 'Female']

    #Preprocess image and prepare it for the classification
    # image: the input image that needs to be preprocessed
    # scalefactor: after mean subtraction image can be scaled by the scalefactorself.
    # size: size that Convolutional Neural Network expects.
    # mean values here are for mean subtraction, usually the values that were used for
    # training are used for classification as well
    mean = (104, 117, 123)
    blob = cv2.dnn.blobFromImage(image, 1.0, (227,227), mean, swapRB=False)

    #Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    return gender

image = cv2.imread("images/sample.jpg")
gender = gender_detector(image)
cv2.imshow("Gender: {}".format(gender),image)
cv2.waitKey()
