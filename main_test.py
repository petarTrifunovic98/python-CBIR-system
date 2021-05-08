from redis_database_access import RedisDatabaseAccessor
from image_processor import ImageProcessor
from image_retrieval_service import ImageRetrievalService
from sorting_strategies import RedisSortStrategy
import cv2 as cv
import numpy as np

DBAccess = RedisDatabaseAccessor('./images', 'localhost', 6379)
image_processor = ImageProcessor(3, [1], [0])
sorting_strategy = RedisSortStrategy('localhost', 6379)

service = ImageRetrievalService('./images', image_processor, DBAccess, sorting_strategy)
service.add_images('./images')

query_img = cv.imread('./images/ruza.jpg')
results = service.get_similar_images(query_img, 5)

print(results)
