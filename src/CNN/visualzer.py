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


visualizer("../../data/females","Female","../../plots/female_CNN.png")
