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
        split_name = image.file_name.split('.')
        extension = split_name[1]
        new_name = str(self.redisDB.incr('naming.counter'))
        new_name += '.'
        new_name += extension
        cv.imwrite(self.dir + '/' + new_name, image.img)
        for el in image.vector:
            self.redisDB.append('vector:' + str(new_name), str(el) + ' ')
        i = 0
        for color in self.img_config['colors']:
            self.redisDB.zadd(color + ':mean', {str(new_name): str(image.discrete_vector[i * 2])})
            self.redisDB.zadd(color + ':std.deviation', {str(new_name): str(image.discrete_vector[i * 2 + 1])})
            i += 1

    def get_similar_images(self, image: Image):
        colors = self.img_config['colors']
        offset = 15
        similar_images_sets = []

        for i in range(len(colors)):
            self.add_similar_images_to_set(colors[i], 'mean', image.get_ith_discrete_mean(i), offset)
            self.add_similar_images_to_set(colors[i], 'std.deviation', image.get_ith_discrete_std_dev(i), offset)
            similar_images_sets.append('similar.images:' + colors[i] + ':mean')
            similar_images_sets.append('similar.images:' + colors[i] + ':std.deviation')

        # similar_images = self.redisDB.smembers('similar.images')
        similar_images = self.redisDB.sinter(similar_images_sets)

        # self.redisDB.delete('similar.images')
        for i in range(len(colors)):
            self.redisDB.delete('similar.images:' + colors[i] + ':mean')
            self.redisDB.delete('similar.images:' + colors[i] + ':std.deviation')
        return similar_images

    def add_similar_images_to_set(self, color, feature, vector_value, offset):
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
            min_rank = rank - offset if (rank - offset) >= 0 else 0
            redis_res = self.redisDB.zrange(color + ':' + feature, min_rank, rank + offset)
            # self.redisDB.sadd('similar.images', *redis_res)
            self.redisDB.sadd('similar.images:' + color + ':' + feature, *redis_res)

    def get_image_vector(self, img_name):
        vector = self.redisDB.get('vector:' + str(img_name)).split()
        vector = [float(string) for string in vector]
        return vector

    def get_image(self, img_name):
        img = cv.imread(self.dir + '/' + img_name)
        return img
