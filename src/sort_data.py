"""File for sorting the data"""

import os
import cv2
from glob import glob
import random

path = "../data/crop_part1"

def preprocess_filenames(path):
    data = os.listdir(path)
    files = []
    for file_name in data:
        files.append(file_name.split('_'))
    return files

processed_files  = preprocess_filenames(path)

def filter_file_names(age, gender, ethnicity):
    filtered = []
    for file_name in processed_files:
        if age != '-' and gender != '-' and ethnicity != '-':
            if file_name[0] == age and file_name[1] == gender and file_name[2] == ethnicity:
                filtered.append('_'.join(file_name))
        if age == '-' and gender == '-':
            if file_name[2] == ethnicity :
                filtered.append('_'.join(file_name))
        if age == '-' and ethnicity == '-':
            if file_name[1] == gender :
                filtered.append('_'.join(file_name))
        if gender == '-' and ethnicity == '-':
            if file_name[0] == age :
                filtered.append('_'.join(file_name))
        if age == '-':
            if file_name[1] == gender and file_name[2] == ethnicity :
                filtered.append('_'.join(file_name))
        if gender == '-':
            if file_name[0] == age and file_name[2] == ethnicity :
                filtered.append('_'.join(file_name))
        if ethnicity == '-':
            if file_name[0] == age and file_name[1] == gender :
                filtered.append('_'.join(file_name))
    return filtered

# Please refer the labels in README.md, if you don't want to use label put dash
#instead of a number
filtered  = filter_file_names('-','0', '4')
random_images = random.sample(filtered, 200)
path_to_classified = '../data/random_200/males/other'
for i in random_images:
    data_path = os.path.join(path,"{}".format(i))
    img = cv2.imread(data_path)
    cv2.imwrite(os.path.join(path_to_classified , "{}".format(i)), img)
