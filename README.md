# facial_recognition_bias

This is a project for showing ethical issues with facial recognition technologies.

# Run

Please refer to the `README.md` for each individual program in `src` directory.

## Data

The images used here are retrieved from [Kaggle](https://www.kaggle.com/) and are stored in file `data/images`. File stores about 10 000 face images. The images are annotated with age, gender and ethnicity. The images are cropped and aligned.

The labels of each face image is embedded in the file name, formatted like
age_gender_race_date&time.jpg

- age is an integer from 0 to 116, indicating the age
- gender is either 0 (male) or 1 (female)
- race is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

More information on data can be found [here](https://susanqq.github.io/UTKFace/).
