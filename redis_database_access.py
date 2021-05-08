import os
import cv2 as cv
import numpy as np
from image_processor import ImageProcessor
from base_database_accessor import BaseDatabaseAccessor
import redis


class RedisDatabaseAccessor(BaseDatabaseAccessor):

    def __init__(self, redis_host, redis_port):
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
                self.redisDB.zadd(colors[i] + ':std.deviation', {str(key): str(discrete_vector[i * 2 + 1])})

    def get_similar(self, query_vector, query_vector_discrete):
        colors = ['R', 'G', 'B']
        offset = 2
        for i in range(len(colors)):
            upper_redis_el = self.redisDB.zrangebyscore(colors[i] + ':mean', query_vector_discrete[i * 2], float('inf'),
                                                        start=0, num=1, withscores=True)
            lower_redis_el = self.redisDB.zrevrangebyscore(colors[i] + ':mean', query_vector_discrete[i * 2],
                                                           float('-inf'), start=0, num=1, withscores=True)
            upper_el_diff = \
                abs(query_vector_discrete[i * 2] - upper_redis_el[0][1]) if len(upper_redis_el) > 0 else float('inf')
            lower_el_diff = \
                abs(query_vector_discrete[i * 2] - lower_redis_el[0][1]) if len(lower_redis_el) > 0 else float('inf')

            if (len(upper_redis_el) > 0) and (len(lower_redis_el) > 0):
                closest_el = upper_redis_el[0][0] if upper_el_diff < lower_el_diff else lower_redis_el[0][0]
                rank = self.redisDB.zrank(colors[i] + ':mean', closest_el)
                min_rank = rank - offset if (rank - offset) >= 0 else 0
                redis_res = self.redisDB.zrange(colors[i] + ':mean', min_rank, rank+offset)
                self.redisDB.sadd('similar.images', *redis_res)

            upper_redis_el = self.redisDB.zrangebyscore(colors[i] + ':std.deviation', query_vector_discrete[i * 2 + 1],
                                                        float('inf'), start=0, num=1, withscores=True)
            lower_redis_el = \
                self.redisDB.zrevrangebyscore(colors[i] + ':std.deviation', query_vector_discrete[i * 2 + 1],
                                              float('-inf'), start=0, num=1, withscores=True)
            upper_el_diff = \
                abs(query_vector_discrete[i * 2] - upper_redis_el[0][1]) if len(upper_redis_el) > 0 else float('inf')
            lower_el_diff = \
                abs(query_vector_discrete[i * 2] - lower_redis_el[0][1]) if len(lower_redis_el) > 0 else float('inf')

            if (len(upper_redis_el) > 0) and (len(lower_redis_el) > 0):
                closest_el = upper_redis_el[0][0] if upper_el_diff < lower_el_diff else lower_redis_el[0][0]
                rank = self.redisDB.zrank(colors[i] + ':std.deviation', closest_el)
                min_rank = rank - offset if (rank - offset) >= 0 else 0
                redis_res = self.redisDB.zrange(colors[i] + ':std.deviation', min_rank, rank + offset)
                self.redisDB.sadd('similar.images', *redis_res)

        similar_images = self.redisDB.smembers('similar.images')

        return similar_images

    def get_vector(self, key):
        vector = self.redisDB.get('vector:' + str(key)).split()
        vector = [float(string) for string in vector]
        return vector


