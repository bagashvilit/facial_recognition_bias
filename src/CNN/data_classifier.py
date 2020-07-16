"""
Classifying dataset based on gender.

The dataset is divided based on race(for more information about races represented here please refer to the README.md).
Each race group has randomly selected 200 images and they are stored in data/random_200.
The ages of people in the images vary from 1 to 100 years old.

The program will plot the percentage of correct guesses as a bar chart, and the
plots will also be saved in plots folder.

If you would like to change the dataset or change path to where plots will be saved,
you may modify the arguments accordinly for visualiser() function

To run the program make sure to call visualizer() function with relevant argumnets,
you can see the suggested calls commented out below. You may use them.

Then navigate to src/CNN and run: python data_classifier.py.
"""

import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Convolution_neural_net


def load_images_from_folder(folder):
    """Read the images from the folder"""
    data_path = os.path.join(folder,"*.jpg")
    filenames = glob(data_path)
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]
    return images

def list_dirs(path):
    list_files = []
    for root, dirs, files in sorted(os.walk(path)):
        list = []
        for dir in dirs:
            list.append(dir)
        return(list)


def visualizer(path,gender,save):
    # path: represents path to the dataset
    # gender: Female/Male
    # save: path to the place where the plots need to be saved
    dirs = (list_dirs(path))
    loaded = []
    for dir in dirs:
        loaded.append(load_images_from_folder("{}/{}".format(path,dir)))

    labeled_all = []

    for file in loaded:
        labeled_per_race = []
        labeled_all.append(labeled_per_race)
        for image in file:
            labeled_per_race.append(Convolution_neural_net.gender_detector(image))

    correct_guess = []
    for i in labeled_all:
        correct_guess.append(((i.count(gender)/len(i))*100))


    y_pos = np.arange(len(dirs))
    plt.bar(y_pos, correct_guess, align='center', alpha=0.5)
    plt.xticks(y_pos, dirs)
    plt.ylabel('Percentage')
    plt.xlabel('Race')
    plt.title("Correct guesses for images of {}s\nConvolution Neural Network".format(gender))
    plt.savefig(save)
    plt.show()

"""plot the graph of correct guesses for females' images for all the race groups represented here."""
# visualizer("../../data/random_200/females","Female","../../plots/CNN_female_200.png")
""" plot the graph of correct guesses for males' images for all the race groups represented here."""
# visualizer("../../data/random_200/males","Male","../../plots/CNN_male_200.png")
