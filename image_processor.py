import cv2 as cv
import numpy as np
import statistics as stat


class ImageProcessor:

    def __init__(self, num_of_colors, discrete_multiplier, discrete_divider):
        self.num_of_colors = num_of_colors
        self.discrete_multiplier = discrete_divider
        self.discrete_divider = discrete_divider

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
        discrete_vector = (vector * self.discrete_multiplier) // self.discrete_multiplier
        return discrete_vector
