# Table of contents

* [About](#about)
* [installation](#installation)
* [info for developers](#info-for-developers)
* [info for users](#info-for-users)
* [testing](#testing)
  + [Automated Testing](#automated-testing)
  + [Code Linting](#code-linting)
* [Run](#run)
* [ethical discussions](#ethical-discussions)
* [future work](#future-work)
* [Contributing](#contributing)
* [Data](#data)

## About

Facial recognition software has become very popular in many industries, including but not limited to law enforcement, airports, healthcare facilities, technology manufacturing companies. These type of technologies let the users track people without their knowledge and gather data about them. As facial recognition software gains popularity some ethical issues are rising regarding the development and use of these tools. According to a report by the [National Institute of Standards and Technology](https://www.nist.gov/news-events/news/2019/12/nist-study-evaluates-effects-race-age-sex-face-recognition-software), comercial facial recognition tools exhibited biases with age, gender and race.This is an important issue because biased facial recognition technology in law enforcment may lead to false accusations and arrests, or in airports it may cause delayed flights.
Therefore the purpose of this project is to highlight ethical issues with face recognition technologies, compare efficiency of different classification algorithms and raise questions about development and use of face recognition tools.

## Installation

- Clone the source code onto your machine

    With HTTPS:

    `https://github.com/Allegheny-Mozilla-Fellows/facial_recognition_bias.git`

    With SSH:

    `git@github.com:Allegheny-Mozilla-Fellows/facial_recognition_bias.git`

## Run

## Development info

## Testing

## Future work

## Data

The images used used in this project are retrieved from [Kaggle](https://www.kaggle.com/) and are stored in file `data/images` directory. File stores about 10 000 face images. The images are annotated with age, gender and ethnicity. The images are cropped and aligned.

The labels of each face image is embedded in the file name, formatted like
age_gender_race_date&time.jpg

- age is an integer from 0 to 116, indicating the age
- gender is either 0 (male) or 1 (female)
- race is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

More information on data can be found [here](https://susanqq.github.io/UTKFace/).

## Reading Material

## Ethical Discussions