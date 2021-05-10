import redis
import json


class BaseSortStrategy:

    def sort(self, dictionary, descending):
        pass


class RedisSortStrategy(BaseSortStrategy):

    def __init__(self, redis_host, redis_port):
        redis_config_file = open('./redis_config.json')
        redis_config = json.load(redis_config_file)
        self.redisDB = redis.Redis(host=redis_config['host'], port=redis_config['port'], db=0, decode_responses=True)

    def sort(self, dictionary, descending):
        for key in dictionary:
            self.redisDB.zadd('sorted.data', {str(key): dictionary[key]})
        sorted_array = \
            self.redisDB.zrevrange('sorted.data', 0, -1) if descending else self.redisDB.zrange('sorted.data', 0, -1)
        self.redisDB.delete('sorted.data')
        return sorted_array
