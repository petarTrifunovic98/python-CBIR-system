import os
import cv2 as cv
import numpy as np
from image_processor import ImageProcessor
import redis


class DatabaseAccessor:

    def __init__(self, images_dir_path, redis_host, redis_port):
        self.dir = images_dir_path
        self.images = None
        self.image_paths = None
        self.vectors = None
        self.texture_vectors = None
        self.image_processor = ImageProcessor(3, 10000, 100, 1, [0])
        self.redisDB = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

    def load_images(self):
        self.images = {}
        for filename in os.listdir(self.dir):
            img = cv.imread(os.path.join(self.dir, filename))
            if img is not None:
                self.images[filename] = img

    def calculate_vectors(self):
        if self.images is None:
            self.load_images()
        self.vectors = {}
        for key in self.images:
            self.vectors[key] = self.image_processor.generate_vector(self.images[key])
            img = cv.cvtColor(self.images[key], cv.COLOR_BGR2GRAY)
            self.vectors[key] = np.concatenate((self.vectors[key], self.image_processor.get_texture_vector(img)))

    def calculate_texture_vectors(self):
        if self.images is None:
            self.load_images()
        self.texture_vectors = {}
        for key in self.images:
            gray_img = cv.cvtColor(self.images[key], cv.COLOR_BGR2GRAY)
            self.texture_vectors[key] = self.image_processor.get_texture_vector(gray_img)

    def load_database(self):
        if self.vectors is None:
            self.calculate_vectors()
        if self.texture_vectors is None:
            self.calculate_texture_vectors()
        for key in self.vectors:
            vector = self.vectors[key]
            for el in vector:
                self.redisDB.append('vector:' + str(key), str(el) + ' ')
            discrete_vector = self.image_processor.make_vector_discrete(vector[0:6])
            # discrete_vector = self.image_processor.make_vector_discrete(vector)
            colors = ['R', 'G', 'B']
            for i in range(len(discrete_vector) // 2):
                # self.redisDB.append(colors[i] + ':mean:' + str(discrete_vector[i * 2]), str(key) + " ")
                self.redisDB.zadd(colors[i] + ':mean', {str(key): str(discrete_vector[i * 2])})
            texture_vector = self.texture_vectors[key]
            for el in texture_vector:
                self.redisDB.append('texture.vector:' + str(key), str(el) + ' ')

    def get_similar_images(self, query_img):
        colors = ['R', 'G', 'B']
        query_vector = self.image_processor.generate_vector(query_img)
        query_vector_discrete = self.image_processor.make_vector_discrete(query_vector)

        query_img_gray = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
        query_vector = np.concatenate((query_vector, self.image_processor.get_texture_vector(query_img_gray)))
        query_vector = np.concatenate(
            (query_vector[0:6], [query_vector[6]], [0], [query_vector[7]], [0], [query_vector[8]], [0]))

        offset = 5

        for i in range(len(query_vector_discrete) // 2):
            redis_res = self.redisDB.zrangebyscore(colors[i] + ":mean", query_vector_discrete[i * 2] - offset,
                                                   query_vector_discrete[i * 2] + offset)
            self.redisDB.sadd(colors[i] + ':similar.images', *redis_res)

        keys = []
        for color in colors:
            keys.append(color + ':similar.images')
        similar_images = self.redisDB.sinter(keys)
        for image in similar_images:
            img_vector = self.redisDB.get('vector:' + str(image)).split()
            img_vector = [float(string) for string in img_vector]
            img_vector = img_vector[0:6] + [img_vector[6]] + [0] + [img_vector[7]] + [0] + [img_vector[8]] + [0]
            distance = self.image_processor.get_manhattan_distance(img_vector, query_vector, 2)
            self.redisDB.zadd('sorted.similar.images', {str(image): distance})

        similar_images = self.redisDB.zrange('sorted.similar.images', 0, -1)
        self.redisDB.delete('sorted.similar.images')

        return similar_images


