import numpy as np
import math
import glob
from torchvision import transforms
from PIL import Image
import torch
from skimage.measure import compare_ssim
import cv2

def cal_psnr(predict,label,data_range):
    predict = np.float64(predict) / data_range
    label = np.float64(label) / data_range
    mse = np.mean(np.square(predict-label))
    if mse == 0:
        return 100
    else:
        PIXEL_MAX = 1
        return 20*math.log10(PIXEL_MAX/math.sqrt(mse))