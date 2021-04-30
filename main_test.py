from database_access import DatabaseAccessor
from image_processor import ImageProcessor
from image_retrieval_service import ImageRetrievalService
import cv2 as cv
import numpy as np

DBAccess = DatabaseAccessor('./images', 'localhost', 6379)
image_processor = ImageProcessor(3, 10000, 100, 1, [0])

service = ImageRetrievalService('./images', image_processor, DBAccess)
service.add_images('./images')

query_img = cv.imread('./images/boat5.jpg')
results = service.get_similar_images(query_img)

print(results)
