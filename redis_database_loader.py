import os
import cv2 as cv
import numpy as np
from image_processor import ImageProcessor
import redis

r = redis.Redis(host='localhost', port=6379, db=0)
#r.set()

class DatabaseLoader:

    def __init__(self, images_dir_path):
        self.dir = images_dir_path
        self.images = None
        self.image_paths = None
        self.vectors = {}
        self.image_processor = ImageProcessor(3, 1000, 100)

    def load_images(self):
        self.images = {}
        for filename in os.listdir(self.dir):
            img = cv.imread(os.path.join(self.dir, filename))
            if img is not None:
                self.images[filename] = img

    def calculate_vectors(self):
        if self.images is None:
            self.load_images()
        for key in self.images:
            self.vectors[key] = self.image_processor.generate_vector(self.images[key])

    def load_database(self):
        discrete_vectors = self.vectors.copy()
        for key in self.vectors:
            discrete_vectors[key] = self.image_processor.make_vector_discrete(self.vectors[key])

