import cv2 as cv
import numpy as np
import statistics as stat
from skimage.feature import greycomatrix, greycoprops
import json
import pywt
import math


class ImageProcessor:

    def __init__(self):
        img_config_file = open('./img_config.json')
        img_config = json.load(img_config_file)
        self.num_of_colors = len(img_config['colors'])
        self.glcm_distances = [img_config['glcm_distance']]
        self.glcm_angles = img_config['glcm_angles']
        self.glcm_props = img_config['glcm_props']
        self.img_size = (img_config['size'][0], img_config['size'][1])
        self.wavelet_type = img_config['wavelet_type']

    def generate_hist_vector(self, image):
        vector = np.empty([6])
        for i in range(self.num_of_colors):
            hist = cv.calcHist([image], [i], None, [256], [0, 256])
            normalized_hist = hist / (image.shape[0] * image.shape[1])
            hist_weighted_elements = [ind * el[0] for ind, el in enumerate(normalized_hist)]
            mean = stat.mean(hist_weighted_elements)
            deviation = stat.pstdev(hist_weighted_elements, mean)
            vector[i * 2] = mean
            vector[i * 2 + 1] = deviation
        return vector

    def generate_glcm_texture_vector(self, image):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        angle_radians = [math.radians(angle) for angle in self.glcm_angles]
        vector = np.empty(3)
        vector[0] = 0
        vector[1] = 0
        vector[2] = 0
        for angle_radian in angle_radians:
            glcm = greycomatrix(image_gray, self.glcm_distances, [angle_radian], levels=256, symmetric=True, normed=True)
            vector[0] += greycoprops(glcm, self.glcm_props[0])[0][0]
            vector[1] += greycoprops(glcm, self.glcm_props[1])[0][0]
            vector[2] += greycoprops(glcm, self.glcm_props[2])[0][0]

        vector_sum = np.sum(vector)
        vector = vector / vector_sum
        return vector

    def generate_wavelet_texture_vector(self, image):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        coefficients = pywt.dwt2(image_gray, self.wavelet_type)
        vector = np.empty(6)
        LL, (LH, HL, HH) = coefficients
        a = np.linalg.norm(LL, axis=0)
        h = np.linalg.norm(LH, axis=0)
        v = np.linalg.norm(HL, axis=1)
        vector[0] = a.mean()
        vector[1] = np.std(a)
        vector[2] = h.mean()
        vector[3] = np.std(h)
        vector[4] = v.mean()
        vector[5] = np.std(v)
        vector_sum = np.sum(vector)
        vector = vector / vector_sum
        return vector

    def resize_image(self, image):
        new_img = cv.resize(image, self.img_size)
        return new_img
