from image_processor import ImageProcessor
from image_repository import ImageRepository
from sorting_strategies import *
from image import Image
import math_helper as mh
import os
import cv2 as cv
import numpy as np


class ImageRetrievalService:

    def __init__(self, image_processor: ImageProcessor, sorting_strategy: BaseSortStrategy,
                 image_rep: ImageRepository):
        self.image_processor = image_processor
        self.sorting_strategy = sorting_strategy
        self.image_repository = image_rep

    def add_images(self, from_dir):
        for filename in os.listdir(from_dir):
            img = cv.imread(os.path.join(from_dir, filename))
            if img is not None:
                img = self.image_processor.resize_image(img)
            if img is not None:
                hist_vector = self.image_processor.generate_hist_vector(img)
                # tex_vector = self.image_processor.generate_glcm_texture_vector(img)
                tex_vector = self.image_processor.generate_wavelet_texture_vector(img)
                vector = np.concatenate((hist_vector, tex_vector))

                vector_sum = np.sum(vector[0:6])
                vector[0:6] = vector[0:6] / vector_sum
                discrete_vector = mh.make_vector_discrete(vector)
                image = Image(filename, from_dir, img, vector, discrete_vector, hist_vector, tex_vector)
                self.image_repository.save_image(image)

    def get_similar_images(self, dir_name, file_name, limit):
        img = cv.imread(dir_name + '/' + file_name)
        img = self.image_processor.resize_image(img)
        query_hist_vector = self.image_processor.generate_hist_vector(img)
        # query_tex_vector = self.image_processor.generate_glcm_texture_vector(img)
        query_tex_vector = self.image_processor.generate_wavelet_texture_vector(img)
        query_vector = np.concatenate((query_hist_vector, query_tex_vector))

        query_vector_sum = np.sum(query_vector[0:6])
        query_vector[0:6] = query_vector[0:6] / query_vector_sum
        query_vector_discrete = mh.make_vector_discrete(query_vector)

        image = Image(file_name, dir_name, img, query_vector, query_vector_discrete, query_hist_vector, query_tex_vector)

        similar = self.image_repository.get_similar_images(image)
        distances = {}
        distances_hist = {}
        distances_tex = {}
        for img_name in similar:
            img_vector = self.image_repository.get_image_vector(img_name)
            img_hist_vector = self.image_repository.get_image_hist_vector(img_name)
            img_tex_vector = self.image_repository.get_image_tex_vector(img_name)
            # distances[img_name] = mh.get_manhattan_distance(img_vector, query_vector)
            # distances[img_name] = mh.get_euclidean_distance(img_vector, query_vector)
            # distances[img_name] = mh.get_cosine_distance(img_vector, query_vector)
            distances_hist[img_name] = mh.get_cosine_distance(img_hist_vector, query_hist_vector)
            distances_tex[img_name] = mh.get_cosine_distance(img_tex_vector, query_tex_vector)
            distance_hist = mh.get_cosine_distance(img_hist_vector, query_hist_vector)
            distance_tex = mh.get_cosine_distance(img_tex_vector, query_tex_vector)

        sorted_similar_names = []
        sorted_similar_images = []
        if len(similar) > 0:
            max_hist = max(distances_hist.values())
            max_tex = max(distances_tex.values())
            if max_hist > 0.0 and max_tex > 0.0:
                distances_hist = {key: value / max_hist for key, value in distances_hist.items()}
                distances_tex = {key: value / max_tex for key, value in distances_tex.items()}
                for img_name in similar:
                    distances[img_name] = distances_hist[img_name] + distances_tex[img_name]
                sorted_similar_names = self.sorting_strategy.sort(distances, False)
                sorted_similar_names = sorted_similar_names[0:limit]
            else:
                sorted_similar_names = similar

        for img_name in sorted_similar_names:
            img = self.image_repository.get_image(img_name)
            image = Image(img_name, '', img, None, None, None, None)
            sorted_similar_images.append(image)

        return sorted_similar_images
