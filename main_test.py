from database_access import DatabaseAccessor
from image_processor import ImageProcessor
import cv2 as cv

DBAccess = DatabaseAccessor('./images', 'localhost', 6379)
img_processor = ImageProcessor(3, 10000, 100)
# DBAccess.load_database()

query_img = cv.imread('./images/boat1.jpg')
query_vector = img_processor.generate_vector(query_img)
query_vector_discrete = img_processor.make_vector_discrete(query_vector)

results = DBAccess.get_similar_images(query_vector_discrete)
print(results)
