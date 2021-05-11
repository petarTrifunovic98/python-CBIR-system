from image_processor import ImageProcessor
from image_retrieval_service import ImageRetrievalService
from sorting_strategies import RedisSortStrategy
from image_repository import ImageRepository
import cv2 as cv
import numpy as np
import json


img_config_file = open('./img_config.json')
img_config = json.load(img_config_file)
w = img_config["size"][0]
h = img_config["size"][1]
display_size = (w, h)
vert_offset = 20
hor_offset = 10


image_processor = ImageProcessor()
sorting_strategy = RedisSortStrategy()
image_repository = ImageRepository('./images')

service = ImageRetrievalService(image_processor, sorting_strategy, image_repository)

while True:
    option = input("Enter 'X' to exit.\nEnter 'A' to add new images to the database.\n"
                   "Enter 'S' to search for similar images:\n")
    if option == 'X':
        break
    elif option == 'A':
        dir_name = input("Enter the path to the directory containing the images you want to add. "
                         "Enter 'X' to cancel:\n")
        if dir_name != 'X':
            service.add_images(dir_name)
    elif option == 'S':
        dir_name = input("Enter the directory (only the directory!) name in which the query image is located. "
                         "Enter 'X' to cancel:\n")
        if dir_name != 'X':
            file_name = input("Enter the filename (only the filename!) of the query image. Enter 'X' to cancel:\n")
            if file_name != 'X':
                results = service.get_similar_images(dir_name, file_name, 5)
                print('Most similar images: ')
                res_img = np.zeros((2 * h + vert_offset, (len(results)) * (w + hor_offset), 3), np.uint8)
                query_img = cv.resize(cv.imread(dir_name + '/' + file_name), display_size)
                res_img[0:h, 0:w] = query_img
                i = 0
                for result in results:
                    print(result.file_name)
                    res_img[h + vert_offset: 2 * h + vert_offset, i * (w + hor_offset):i * (w + hor_offset) + w] = \
                        cv.resize(result.img, display_size)
                    i += 1
                cv.imshow('res', res_img)
                cv.waitKey(0)
                cv.destroyWindow('res')
print("Goodbye!")
