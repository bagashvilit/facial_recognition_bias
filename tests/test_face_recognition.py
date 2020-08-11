import pickle

import cv2
import joblib
import pytest
import sklearn

from pyimagesearch.rgbhistogram import RGBHistogram


@pytest.mark.parametrize(
    "input_image,expected_gender",
    [("tests/images/17_1_0.jpg", 1), ("tests/images/23_1_2.jpg", 1)],
)
def test_SVM(input_image, expected_gender):
    desc = RGBHistogram([8, 8, 8])

    model = pickle.load(open("src/SVM/SVM_model.pkl", "rb"))

    image = cv2.imread(input_image)

    features = desc.describe(image)
    gender = (model.predict([features.flatten()]))[0]
    assert gender == expected_gender


@pytest.mark.parametrize(
    "input_image,expected_gender",
    [("tests/images/17_1_0.jpg", 1), ("tests/images/23_1_2.jpg", 1)],
)
def test_Random_Forest(input_image, expected_gender):
    desc = RGBHistogram([8, 8, 8])

    model = pickle.load(open("src/RandomForest/RandomForest_model.pkl", "rb"))

    image = cv2.imread(input_image)

    features = desc.describe(image)
    gender = (model.predict([features.flatten()]))[0]
    assert gender == expected_gender
