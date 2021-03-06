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
                 image_rep: ImageRepository, use_wavelets):
        self.image_processor = image_processor
        self.sorting_strategy = sorting_strategy
        self.image_repository = image_rep
        img_config_file = open('./img_config.json')
        img_config = json.load(img_config_file)
        if use_wavelets:
            self.texture_props = img_config['wavelet_props']
        else:
            self.texture_props = img_config['glcm_props']
        self.use_wavelets = use_wavelets
        general_config_file = open('./general_config.json')
        general_config = json.load(general_config_file)
        self.distance_limit_color = general_config['distance_limit_color']
        self.distance_limit_tex = general_config['distance_limit_tex']
        self.distance_limit_sum = general_config['distance_limit_sum']

    def add_images(self, from_dir):
        for filename in os.listdir(from_dir):
            img = cv.imread(os.path.join(from_dir, filename))
            if img is not None:
                img = self.image_processor.resize_image(img)
            if img is not None:
                hist_vector = self.image_processor.generate_hist_vector(img)
                if self.use_wavelets is True:
                    tex_vector = self.image_processor.generate_wavelet_texture_vector(img)
                else:
                    tex_vector = self.image_processor.generate_glcm_texture_vector(img)

                vector = np.concatenate((hist_vector, tex_vector))

                vector_sum = np.sum(vector[0:6])
                vector[0:6] = vector[0:6] / vector_sum
                discrete_vector = mh.make_vector_discrete(vector)
                image = Image(filename, from_dir, img, vector, discrete_vector, hist_vector, tex_vector)
                self.image_repository.save_image(image, self.texture_props)

    def get_similar_images(self, dir_name, file_name, limit):
        img = cv.imread(dir_name + '/' + file_name)
        img = self.image_processor.resize_image(img)
        query_hist_vector = self.image_processor.generate_hist_vector(img)
        if self.use_wavelets is True:
            query_tex_vector = self.image_processor.generate_wavelet_texture_vector(img)
        else:
            query_tex_vector = self.image_processor.generate_glcm_texture_vector(img)

        query_vector = np.concatenate((query_hist_vector, query_tex_vector))

        query_vector_sum = np.sum(query_vector[0:6])
        query_vector[0:6] = query_vector[0:6] / query_vector_sum
        query_vector_discrete = mh.make_vector_discrete(query_vector)

        image = Image(file_name, dir_name, img, query_vector, query_vector_discrete, query_hist_vector, query_tex_vector)

        similar = self.image_repository.get_similar_images(image, self.texture_props)
        distances = {}
        distances_sum = {}
        distances_hist = {}
        distances_tex = {}
        for img_name in similar:
            img_hist_vector = self.image_repository.get_image_hist_vector(img_name)
            img_tex_vector = self.image_repository.get_image_tex_vector(img_name)
            distances_hist[img_name] = mh.get_cosine_distance(img_hist_vector, query_hist_vector)
            distances_tex[img_name] = mh.get_cosine_distance(img_tex_vector, query_tex_vector)
            distances_sum[img_name] = distances_hist[img_name] + distances_tex[img_name]

        sorted_similar_names = []
        sorted_similar_images = []
        if len(similar) > 0:
            max_hist = max(distances_hist.values())
            max_tex = max(distances_tex.values())
            max_sum = max(distances_sum.values())
            if max_sum > 0.0:
                # distances_hist = {key: value / max_hist for key, value in distances_hist.items()}
                # distances_tex = {key: value / max_tex for key, value in distances_tex.items()}
                distances_sum = {key: value / max_sum for key, value in distances_sum.items()}
                for img_name in similar:
                    if distances_sum[img_name] <= self.distance_limit_sum:
                        distances[img_name] = distances_sum[img_name]
                sorted_similar_names = self.sorting_strategy.sort(distances, False)
                sorted_similar_names = sorted_similar_names[0:limit]
            else:
                sorted_similar_names = similar

        for img_name in sorted_similar_names:
            img = self.image_repository.get_image(img_name)
            image = Image(img_name, '', img, None, None, None, None)
            sorted_similar_images.append(image)

        return sorted_similar_images
