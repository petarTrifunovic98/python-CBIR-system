import os
import cv2 as cv
import numpy as np
from image_processor import ImageProcessor
from base_database_accessor import BaseDatabaseAccessor
import redis


class RedisDatabaseAccessor(BaseDatabaseAccessor):

    def __init__(self, images_dir_path, redis_host, redis_port):
        self.redisDB = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

    def load_database(self, vectors, discrete_vectors):
        for key in vectors:
            vector = vectors[key]
            for el in vector:
                self.redisDB.append('vector:' + str(key), str(el) + ' ')
            discrete_vector = discrete_vectors[key]
            colors = ['R', 'G', 'B']
            for i in range(len(colors)):
                self.redisDB.zadd(colors[i] + ':mean', {str(key): str(discrete_vector[i * 2])})
                self.redisDB.zadd(colors[i] + ':std.deviation:', {str(key): str(discrete_vector[i * 2 + 1])})

    def get_similar(self, query_vector, query_vector_discrete):
        colors = ['R', 'G', 'B']
        offset = 8
        for i in range(len(colors)):
            redis_res = self.redisDB.zrangebyscore(colors[i] + ':mean', query_vector_discrete[i * 2] - offset,
                                                   query_vector_discrete[i * 2] + offset)
            self.redisDB.sadd(colors[i] + ':mean:similar.images', *redis_res)
            redis_res = self.redisDB.zrangebyscore(colors[i] + ':std.deviation:',
                                                   query_vector_discrete[i * 2 + 1] - offset,
                                                   query_vector_discrete[i * 2 + 1] + offset)
            self.redisDB.sadd(colors[i] + ':std.deviation:similar.images', *redis_res)

        keys = []
        for color in colors:
            keys.append(color + ':mean:similar.images')
            keys.append(color + ':std.deviation:similar.images')
        similar_images = self.redisDB.sinter(keys)

        return similar_images

    def get_vector(self, key):
        vector = self.redisDB.get('vector:' + str(key)).split()
        vector = [float(string) for string in vector]
        return vector


