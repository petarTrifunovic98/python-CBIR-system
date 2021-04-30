from database_access import DatabaseAccessor
from image_processor import ImageProcessor
import cv2 as cv
import numpy as np

DBAccess = DatabaseAccessor('./images', 'localhost', 6379)
DBAccess.load_database()

query_img = cv.imread('./images/boat5.jpg')
results = DBAccess.get_similar_images(query_img)

print(results)
