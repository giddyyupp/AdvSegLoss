import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import math


def hypotenuse(x1, y1, x2, y2):
    xSquare = math.pow(x1 - x2, 2)
    ySquare = math.pow(y1 - y2, 2)
    return np.sqrt(xSquare + ySquare)


def generateGuassianKernel2(dimension=9, sigma=3.0):

    kernel = np.zeros((dimension, dimension), dtype=np.float32)
    twoSigmaSquare = 2 * sigma * sigma
    centre = (dimension - 1) / 2

    for i in range(dimension):
        for j in range(dimension):
            distance = hypotenuse(i, j, centre, centre)

            gaussian = (1 / np.sqrt(
                math.pi * twoSigmaSquare
            )) * np.exp((-1) * (math.pow(distance, 2) / twoSigmaSquare))

            kernel[i,j] = gaussian

    return kernel / kernel.sum()


def calculateKernelSize(sigma):
  return int(np.maximum(np.round(sigma * 3) * 2 + 1, 3))


input_path = '/media/test/Samhi/GANILLA/fpn-gan/dataset/ade_20k/trainA'
output_path = '/media/test/Samhi/GANILLA/fpn-gan/dataset/ade_20k/trainB_dog'

if not os.path.exists(output_path):
    os.makedirs(output_path)

im_w = 256
# params
Sigma1 = 1.8 #1.37
Sigma2 = 2.6 #2.37
threshold = 0.420 #0.408

ks1 = calculateKernelSize(Sigma1)
ks2 = calculateKernelSize(Sigma2)

filter_w1 = torch.from_numpy(generateGuassianKernel2(ks1, Sigma1)).unsqueeze(0).unsqueeze(0)
filter_w2 = torch.from_numpy(generateGuassianKernel2(ks2, Sigma2)).unsqueeze(0).unsqueeze(0)

conv1 = nn.Conv2d(1, 1, kernel_size=ks1, stride=1, padding=int(ks1/2), bias=False)
conv2 = nn.Conv2d(1, 1, kernel_size=ks2, stride=1, padding=int(ks2/2), bias=False)

W1 = nn.Parameter(filter_w1)
W2 = nn.Parameter(filter_w2)
W1.requires_grad = False
W2.requires_grad = False
with torch.no_grad():
    conv1.weight = W1
    conv2.weight = W2

for ff in os.listdir(input_path):
    print(ff)
    im_file = os.path.join(input_path, ff)
    img = cv2.imread(im_file)
    img = cv2.resize(img, dsize=(im_w, im_w))
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_tensor = torch.from_numpy(im).float().unsqueeze(0).unsqueeze(0)

    im1_tensor = conv1(im_tensor)
    im2_tensor = conv2(im_tensor)

    diff = im1_tensor - im2_tensor

    diffPositive = diff - torch.min(diff)
    relativeDiff = diffPositive / torch.max(diffPositive)

    temp = relativeDiff - threshold
    temp[temp > 0] = 1.
    temp[temp <= 0] = 0.
    result = torch.mul(temp, 255)

    res = result.cpu().numpy()[0,0,:,:]

    cv2.imwrite(os.path.join(output_path, ff), res)