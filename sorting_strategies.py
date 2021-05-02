import redis


class BaseStrategy:

    def sort(self, dictionary, descending):
        pass


class RedisSortStrategy(BaseStrategy):

    def __init__(self, redis_host, redis_port):
        self.redisDB = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)

    def sort(self, dictionary, descending):
        for key in dictionary:
            self.redisDB.zadd('sorted.data', {str(key): dictionary[key]})
        sorted_array = \
            self.redisDB.zrevrange('sorted.data', 0, -1) if descending else self.redisDB.zrange('sorted.data', 0, -1)
        return sorted_array
