# Binary logistic regression using MNIST dataset

## Overview
Use of two-class logistic regression to classify handwritten digits from the MNIST (Modified National Institute of Standards and Technology) dataset. This dataset contains 60,000 training and 10,000 test images, with each grayscale image having 28x28 pixels.

First, the user selects two digits for classification. Then, a simple logistic binary regression model is implemented in Keras, using sigmoid activation function with gradient descent and cross-entropy loss function.

Accuracy on test dataset is close to 100%, regardless of the set of digits selected by the user.

## Dependencies
numpy==1.14.0, matplotlib==2.0.0, Keras==2.1.3, TensorFlow==1.4.0