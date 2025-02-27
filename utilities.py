import numpy as np
import cv2
# for a given image let's define a function that will return the measurement frame using a random mask and and a white gaussian noise
# y = H*x + n


def preprocess_image(image, size):
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    return image


def measurement_frame(x, H, sigma):
    n = np.random.normal(0, sigma, H.shape[0])
    y = np.dot(H, x) + n
    return y