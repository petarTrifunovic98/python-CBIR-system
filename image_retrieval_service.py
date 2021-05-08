from image_processor import ImageProcessor
from base_database_accessor import BaseDatabaseAccessor
from sorting_strategies import *
import math_helper as mh
import os
import cv2 as cv
import numpy as np


class ImageRetrievalService:

    def __init__(self, images_dir_path, image_processor: ImageProcessor, database_accessor: BaseDatabaseAccessor,
                 sorting_strategy: BaseSortStrategy):
        self.dir = images_dir_path
        self.image_processor = image_processor
        self.database_accessor = database_accessor
        self.sorting_strategy = sorting_strategy

    def add_images(self, from_dir):
        vectors = {}
        discrete_vectors = {}
        for filename in os.listdir(from_dir):
            img = cv.imread(os.path.join(from_dir, filename))
            if img is not None:
                vector = self.image_processor.generate_hist_vector(img)
                #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                vector = np.concatenate((vector, self.image_processor.generate_texture_vector(img)))
                vectors[from_dir + '/' + filename] = vector
                discrete_vectors[from_dir + '/' + filename] = mh.make_vector_discrete(vector[0:6])
        self.database_accessor.load_database(vectors, discrete_vectors)

    def get_similar_images(self, query_img, limit):
        query_vector = self.image_processor.generate_hist_vector(query_img)
        query_vector_discrete = mh.make_vector_discrete(query_vector)
        query_vector = np.concatenate((query_vector, self.image_processor.generate_texture_vector(query_img)))
        #query_vector = np.concatenate((query_vector[0:6], [query_vector[6]], [0], [query_vector[7]], [0],
         #                              [query_vector[8]], [0]))

        similar = self.database_accessor.get_similar(query_vector, query_vector_discrete)
        distances = {}
        for path in similar:
            img_vector = self.database_accessor.get_vector(path)
            path_split = path.split('/')
            name = path_split[len(path_split) - 1]
            distances[name] = mh.get_manhattan_distance(img_vector, query_vector, 1)

        sorted_similar = self.sorting_strategy.sort(distances, False)
        sorted_similar = sorted_similar[0:limit]
        return sorted_similar

