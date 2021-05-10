import cv2 as cv
from image import Image
import redis
import json


class ImageRepository:

    def __init__(self, images_dir_path):
        self.dir = images_dir_path
        img_config_file = open('./img_config.json')
        self.img_config = json.load(img_config_file)
        redis_config_file = open('./redis_config.json')
        redis_config = json.load(redis_config_file)
        self.redisDB = redis.Redis(host=redis_config['host'], port=redis_config['port'], db=0, decode_responses=True)

    def save_image(self, image: Image):
        cv.imwrite(self.dir + '/' + image.file_name, image.img)
        key_part = image.dir_name + '/' + image.file_name
        for el in image.vector:
            self.redisDB.append('vector:' + str(key_part), str(el) + ' ')
        i = 0
        for color in self.img_config['colors']:
            self.redisDB.zadd(color + ':mean', {str(key_part): str(image.discrete_vector[i * 2])})
            self.redisDB.zadd(color + ':std.deviation', {str(key_part): str(image.discrete_vector[i * 2 + 1])})
            i += 1

    def get_similar_images(self, image: Image):
        colors = self.img_config['colors']
        offset = 2
        for i in range(len(colors)):
            self.add_similar_images_to_set(colors[i], 'mean', image.get_ith_discrete_mean(i), offset)
            self.add_similar_images_to_set(colors[i], 'std.deviation', image.get_ith_discrete_std_dev(i), offset)

        similar_images = self.redisDB.smembers('similar.images')
        return similar_images

    def add_similar_images_to_set(self, color, feature, vector_value, range):
        upper_redis_el = self.redisDB.zrangebyscore(color + ':' + feature, vector_value, float('inf'),
                                                    start=0, num=1, withscores=True)
        lower_redis_el = self.redisDB.zrevrangebyscore(color + ':' + feature, vector_value, float('-inf'),
                                                       start=0, num=1, withscores=True)

        upper_el_diff = \
            abs(vector_value - upper_redis_el[0][1]) if len(upper_redis_el) > 0 else float('inf')
        lower_el_diff = \
            abs(vector_value - lower_redis_el[0][1]) if len(lower_redis_el) > 0 else float('inf')

        if (len(upper_redis_el) > 0) and (len(lower_redis_el) > 0):
            closest_el = upper_redis_el[0][0] if upper_el_diff < lower_el_diff else lower_redis_el[0][0]
            rank = self.redisDB.zrank(color + ':' + feature, closest_el)
            min_rank = rank - range if (rank - range) >= 0 else 0
            redis_res = self.redisDB.zrange(color + ':' + feature, min_rank, rank + range)
            self.redisDB.sadd('similar.images', *redis_res)

    def get_image_vector(self, full_path):
        vector = self.redisDB.get('vector:' + str(full_path)).split()
        vector = [float(string) for string in vector]
        return vector
