from image_processor import ImageProcessor
from image_retrieval_service import ImageRetrievalService
from sorting_strategies import RedisSortStrategy
from image_repository import ImageRepository

image_processor = ImageProcessor(3, [1], [0])
sorting_strategy = RedisSortStrategy('localhost', 6379)
image_repository = ImageRepository('./images')

service = ImageRetrievalService('./images', image_processor, sorting_strategy, image_repository)
service.add_images('../images')

results = service.get_similar_images('../images', 'boat1.jpg', 5)

print(results)
