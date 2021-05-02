import cv2 as cv
import numpy as np
import statistics as stat
from skimage.feature import greycomatrix, greycoprops
from skimage import data


class ImageProcessor:

    def __init__(self, num_of_colors, discrete_multiplier, discrete_divider, glcm_distance, glcm_angles):
        self.num_of_colors = num_of_colors
        self.discrete_multiplier = discrete_multiplier
        self.discrete_divider = discrete_divider
        self.glcm_distance = [glcm_distance]
        self.glcm_angles = glcm_angles

    def generate_vector(self, image):
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

    def make_vector_discrete(self, vector):
        discrete_vector = (vector * self.discrete_multiplier) // self.discrete_divider
        return discrete_vector

    def get_manhattan_distance(self, vector1, vector2, step):
        distance = 0
        # for i in range(self.num_of_colors):
        for i in range(len(vector1)):
            distance += abs(vector1[i * step] - vector2[i * step])
        return distance

    def get_texture_vector(self, image):
        glcm = greycomatrix(image, self.glcm_distance, self.glcm_angles, levels=256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')
        energy = greycoprops(glcm, 'correlation')
        inverse_difference = greycoprops(glcm, 'homogeneity')
        vector = np.empty(3)
        vector[0] = contrast[0][0]
        vector[1] = energy[0][0]
        vector[2] = inverse_difference[0][0]
        return vector

