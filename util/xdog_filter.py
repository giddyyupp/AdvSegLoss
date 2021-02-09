import sys

import cv2
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter


def xdog(image, epsilon=0.01):
    """
    Computes the eXtended Difference of Gaussians (XDoG) for a given image. This
    is done by taking the regular Difference of Gaussians, thresholding it
    at some value, and applying the hypertangent function the the unthresholded
    values.
    image: an n x m single channel matrix.
    epsilon: the offset value when computing the hypertangent.
    returns: an n x m single channel matrix representing the XDoG.
    """
    phi = 10

    difference = dog(image, 200, 0.98) / 255
    diff = difference * image

    for i in range(0, len(difference)):
        for j in range(0, len(difference[0])):
            if difference[i][j] >= epsilon:
                difference[i][j] = 1
            else:
                ht = np.tanh(phi * (difference[i][j] - epsilon))
                difference[i][j] = 1 + ht

    return difference * 255


def dog(image, k=200, gamma=1):
    """
    Computes the Difference of Gaussians (DoG) for a given image. Returns an image
    that results from computing the DoG.
    image: an n x m array for which the DoG is computed.
    k: the multiplier the the second Gaussian sigma value.
    gamma: the multiplier for the second Gaussian result.

    return: an n x m array representing the DoG
    """

    s1 = 0.5
    s2 = s1 * k

    gauss1 = gaussian_filter(image, s1)
    gauss2 = gamma * gaussian_filter(image, s2)

    differenceGauss = gauss1 - gauss2
    return differenceGauss


if __name__ == '__main__':
    im_w = 256

    input_path = '/media/test/Samhi/GANILLA/fpn-gan/dataset/ade_20k/trainA'
    output_path = '/media/test/Samhi/GANILLA/fpn-gan/dataset/ade_20k/trainB_xdog2'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for ff in os.listdir(input_path):
        print(ff)
        im_file = os.path.join(input_path, ff)
        img = cv2.imread(im_file)
        img = cv2.resize(img, dsize=(im_w, im_w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Main method defaults to computiong the eXtended Diference of Gaussians
        result = xdog(img)

        cv2.imwrite(os.path.join(output_path, ff), result)

