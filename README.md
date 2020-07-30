# facial_recognition_bias

Project for showing ethical issues with facial recognition technologies

## Tools and libraries

If you have OpenCV and sklearn installed locally on your machine, you can run Python programs directly
from the terminal inside the `src` directory.

## Run

- run classifier with `random forest` inside `src/RandomForest` directory: `python classify.py`

  arguments:
    - `--females` number of images of females, maximum amount is 5407.
    - `--males` number of images of males, maximum amount is 4372.

  sample command: `python classify.py --females 1500 --males 1500`

  You may adjust the numbers as necessary to experiment with the algorithm. The input data
  will be split into training and test data, where test data share is 0.3.

- run classifier with pre-trained `Convolution Neural Network` inside `src/CNN` directory: `python classify.py`

  arguments:
  - `--females` number of images of females, maximum amount is 5407.
  - `--males` number of images of males, maximum amount is 4372.

  sample command:
  `python classify.py --females 1500 --males 1500`

  You may adjust the numbers as necessary to experiment with the classifier. It may
  take couple of minutes as you increase the input data size

## Output

The output is a classification report, and labeled sample image

## Data

The images used here are retrieved from [Kaggle](https://www.kaggle.com/) and are stored in file `data/images`. File stores about 10 000 face images. The images are annotated with age, gender and ethnicity. The images are cropped and aligned.

The labels of each face image is embedded in the file name, formatted like
age_gender_race_date&time.jpg

- age is an integer from 0 to 116, indicating the age
- gender is either 0 (male) or 1 (female)
- race is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

More information on data can be found [here](https://susanqq.github.io/UTKFace/).
