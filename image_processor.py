import cv2 as cv
import numpy as np
import statistics as stat
from skimage.feature import greycomatrix, greycoprops


class ImageProcessor:

    def __init__(self, num_of_colors, glcm_distances, glcm_angles):
        self.num_of_colors = num_of_colors
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles

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

    def generate_texture_vector(self, image):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        glcm = greycomatrix(image_gray, self.glcm_distances, self.glcm_angles, levels=256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'energy')
        #contrast = greycoprops(glcm, 'contrast')
        energy = greycoprops(glcm, 'correlation')
        inverse_difference = greycoprops(glcm, 'homogeneity')
        vector = np.empty(3)
        vector[0] = contrast[0][0]
        vector[1] = energy[0][0]
        vector[2] = inverse_difference[0][0]
        return vector

