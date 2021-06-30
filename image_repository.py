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
        for el in image.hist_vector:
            self.redisDB.append('hist.vector:' + str(new_name), str(el) + ' ')
        for el in image.tex_vector:
            self.redisDB.append('tex.vector:' + str(new_name), str(el) + ' ')
        i = 0
        for color in self.img_config['colors']:
            self.redisDB.zadd(color + ':mean', {str(new_name): str(image.discrete_vector[i * 2])})
            self.redisDB.zadd(color + ':std.deviation', {str(new_name): str(image.discrete_vector[i * 2 + 1])})
            i += 1

        # texture_props = self.img_config['glcm_props']
        texture_props = self.img_config['wavelet_props']

        for i in range(1, len(texture_props) + 1):
            self.redisDB.zadd(texture_props[-i], {str(new_name): str(image.discrete_vector[-i])})

    def get_similar_images(self, image: Image):
        colors = self.img_config['colors']
        # texture_props = self.img_config['glcm_props']
        texture_props = self.img_config['wavelet_props']
        offset = 6
        similar_images_sets = []

        for i in range(len(colors)):
            # self.add_similar_images_to_set(colors[i], 'mean', image.get_ith_discrete_mean(i), offset)
            # self.add_similar_images_to_set(colors[i], 'std.deviation', image.get_ith_discrete_std_dev(i), offset)
            self.add_similar_images_to_set_by_score(colors[i] + ':mean', image.get_ith_discrete_mean(i), offset)
            self.add_similar_images_to_set_by_score(colors[i] + ':std.deviation', image.get_ith_discrete_std_dev(i), offset)
            similar_images_sets.append('similar.images:' + colors[i] + ':mean')
            similar_images_sets.append('similar.images:' + colors[i] + ':std.deviation')

        for i in range(len(texture_props)):
            self.add_similar_images_to_set_by_score(texture_props[i], image.get_ith_discrete_tex_prop(i), offset)
            similar_images_sets.append('similar.images:' + texture_props[i])

        # similar_images = self.redisDB.smembers('similar.images')
        similar_images = self.redisDB.sinter(similar_images_sets)

        # self.redisDB.delete('similar.images')
        for set_name in similar_images_sets:
            self.redisDB.delete(set_name)
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

        if (len(upper_redis_el) >= 0) and (len(lower_redis_el) >= 0):
            closest_el = upper_redis_el[0][0] if upper_el_diff < lower_el_diff else lower_redis_el[0][0]
            rank = self.redisDB.zrank(color + ':' + feature, closest_el)
            min_rank = rank - offset if (rank - offset) >= 0 else 0
            redis_res = self.redisDB.zrange(color + ':' + feature, min_rank, rank + offset)
            # self.redisDB.sadd('similar.images', *redis_res)
            self.redisDB.sadd('similar.images:' + color + ':' + feature, *redis_res)

    def add_similar_images_to_set_by_score(self, key, score, offset):
        redis_res = self.redisDB.zrangebyscore(key, score - offset, score + offset)
        self.redisDB.sadd('similar.images:' + key, *redis_res)

    def get_image_vector(self, img_name):
        vector = self.redisDB.get('vector:' + str(img_name)).split()
        vector = [float(string) for string in vector]
        return vector

    def get_image_hist_vector(self, img_name):
        hist_vector = self.redisDB.get('hist.vector:' + str(img_name)).split()
        hist_vector = [float(string) for string in hist_vector]
        return hist_vector

    def get_image_tex_vector(self, img_name):
        tex_vector = self.redisDB.get('tex.vector:' + str(img_name)).split()
        tex_vector = [float(string) for string in tex_vector]
        return tex_vector

    def get_image(self, img_name):
        img = cv.imread(self.dir + '/' + img_name)
        return img
