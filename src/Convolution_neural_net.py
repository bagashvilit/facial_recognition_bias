import cv2

def load_caffe_models():
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(gender_net)

def gender_detector(gender_net,image):

    #Preprocess image and prepare it for the classification
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)

    #Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    print("Gender : " + gender)

gender_net = load_caffe_models()

video_detector(gender_net)
