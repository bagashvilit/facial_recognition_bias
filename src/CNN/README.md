https://www.tensorflow.org/tutorials/images/cnn
## Tools and libraries

If you have OpenCV and sklearn installed locally on your machine, you can run Python programs directly
from the terminal inside the `src` directory.

## Run

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
