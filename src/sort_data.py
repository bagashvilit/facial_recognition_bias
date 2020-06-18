"""File for sorting the data"""

import os
import cv2

path = "../data/crop_part1"


def preprocess_filenames(path):
    data = os.listdir(path)
    files = []
    for file_name in data:
        files.append(file_name.split('_'))
    return files

processed_files  = preprocess_filenames(path)

def filter_data(age, gender, ethnicity):
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

# Please refer see the labels in README.md, if you don't wan to use label put dash
#instead of a number
filtered  = filter_data('-','1', '2')

print(filtered)
