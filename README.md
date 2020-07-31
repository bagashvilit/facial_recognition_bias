# Table of contents

* [About](#about)
* [installation](#installation)
* [info for developers](#info-for-developers)
* [info for users](#info-for-users)
* [testing](#testing)
  + [Automated Testing](#automated-testing)
  + [Test Coverage](#test-coverage)
  + [Testing with Multiple Python Versions](#testing-with-multiple-python-versions)
  + [Code Linting](#code-linting)
* [Run](#run)
* [ethical discussions](#ethical-discussions)
* [future work](#future-work)
* [Contributing](#contributing)
* [Data](#data)

# About

# installation

# info for developers

# info for users

# Run

# testing

# ethical discussions

# future work

## Data

The images used used in this project are retrieved from [Kaggle](https://www.kaggle.com/) and are stored in file `data/images` directory. File stores about 10 000 face images. The images are annotated with age, gender and ethnicity. The images are cropped and aligned.

The labels of each face image is embedded in the file name, formatted like
age_gender_race_date&time.jpg

- age is an integer from 0 to 116, indicating the age
- gender is either 0 (male) or 1 (female)
- race is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

More information on data can be found [here](https://susanqq.github.io/UTKFace/).
