import cv2 as cv
import numpy as np
import statistics as stat
from skimage.feature import greycomatrix, greycoprops
import json
import pywt


class ImageProcessor:

    def __init__(self):
        img_config_file = open('./img_config.json')
        img_config = json.load(img_config_file)
        self.num_of_colors = len(img_config['colors'])
        self.glcm_distances = [img_config['glcm_distance']]
        self.glcm_angles = [img_config['glcm_angle']]
        self.img_size = (img_config['size'][0], img_config['size'][1])

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
        glcm = greycomatrix(image_gray, self.glcm_distances, self.glcm_angles, levels=256, symmetric=True, normed=True)
        energy = greycoprops(glcm, 'energy')
        correlation = greycoprops(glcm, 'correlation')
        inverse_difference = greycoprops(glcm, 'homogeneity')
        vector = np.empty(3)
        vector[0] = energy[0][0]
        vector[1] = correlation[0][0]
        vector[2] = inverse_difference[0][0]
        return vector

    @staticmethod
    def generate_wavelet_texture_vector(image):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # coefficients = pywt.dwt2(image_gray, 'bior1.3')
        coefficients = pywt.wavedec2(image_gray, 'bior1.3', mode='symmetric', level=2)
        vector = np.empty(2)
        energy = greycoprops(coefficients[0], 'energy')
        vector[0] = coefficients[0].mean()
        vector[1] = np.std(coefficients[0])
        vector_sum = np.sum(vector)
        vector = vector / vector_sum
        return vector

    def resize_image(self, image):
        new_img = cv.resize(image, self.img_size)
        return new_img
