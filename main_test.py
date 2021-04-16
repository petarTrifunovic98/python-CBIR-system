from database_access import DatabaseAccessor
from image_processor import ImageProcessor
import cv2 as cv
import numpy as np

DBAccess = DatabaseAccessor('./images', 'localhost', 6379)
img_processor = ImageProcessor(3, 10000, 100, 1, [0])
DBAccess.load_database()

query_img = cv.imread('./images/boat5.jpg')
query_vector = img_processor.generate_vector(query_img)
query_vector_discrete = img_processor.make_vector_discrete(query_vector)

query_img_gray = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
query_vector = np.concatenate((query_vector, img_processor.get_texture_vector(query_img_gray)))
query_vector = np.concatenate((query_vector[0:6], [query_vector[6]], [0], [query_vector[7]], [0], [query_vector[8]], [0]))

results = DBAccess.get_similar_images(query_vector_discrete, query_vector)
print(results)
