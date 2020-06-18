# facial_recognition_bias

Project for showing biases in facial recognition technologies

## Data

The data is retrieved from [Kaggle](https://www.kaggle.com/) and is stored in file `crop_part1` in folder `data`. File stores over 20,000 face images. The images are annotated with age, gender and ethnicity. The images are cropped and aligned.

## Labels

The labels of each face image is embedded in the file name, formated like
age_gender_race_date&time.jpg

- age is an integer from 0 to 116, indicating the age
- gender is either 0 (male) or 1 (female)
- race is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

More information on data can be found [here](https://susanqq.github.io/UTKFace/).
