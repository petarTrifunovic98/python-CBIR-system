from image_processor import ImageProcessor
from database_access import DatabaseAccessor
import os
import cv2 as cv
import numpy as np


class ImageRetrievalService:

    def __init__(self, images_dir_path, image_processor: ImageProcessor, database_accessor: DatabaseAccessor):
        self.dir = images_dir_path
        self.image_processor = image_processor
        self.database_accessor = database_accessor

    def add_images(self, from_dir):
        vectors = {}
        discrete_vectors = {}
        for filename in os.listdir(from_dir):
            img = cv.imread(os.path.join(self.dir, filename))
            if img is not None:
                vector = self.image_processor.generate_vector(img)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                vector = np.concatenate((vector, self.image_processor.get_texture_vector(img)))
                vectors[filename] = vector
                discrete_vectors[filename] = self.image_processor.make_vector_discrete(vector[0:6])
        self.database_accessor.load_database(vectors, discrete_vectors)

    def get_similar_images(self, query_img):
        query_vector = self.image_processor.generate_vector(query_img)
        query_vector_discrete = self.image_processor.make_vector_discrete(query_vector)
        query_img_gray = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
        query_vector = np.concatenate((query_vector, self.image_processor.get_texture_vector(query_img_gray)))
        query_vector = np.concatenate((query_vector[0:6], [query_vector[6]], [0], [query_vector[7]], [0],
                                       [query_vector[8]], [0]))

        similar = self.database_accessor.get_similar(query_vector, query_vector_discrete)
        return similar

