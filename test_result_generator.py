from image_processor import ImageProcessor
from image_retrieval_service import ImageRetrievalService
from sorting_strategies import RedisSortStrategy
from image_repository import ImageRepository
import cv2 as cv
import numpy as np
import json
import os


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

in_dir_name = input("Enter the path to a directory. All the images in that directory will be used as input images.\n")
out_dir_name = input("Enter the path to the directory where you wish the test results to be saved.\n")
name_addition = input("Enter the string that you wish to append to all the input image names, in order to create the "
                      "output names for them.\n")

for filename in os.listdir(in_dir_name):
    results = service.get_similar_images(in_dir_name, filename, 5)
    res_img = np.zeros((2 * h + vert_offset, (len(results)) * (w + hor_offset), 3), np.uint8)
    query_img = cv.resize(cv.imread(in_dir_name + '/' + filename), display_size)
    res_img[0:h, 0:w] = query_img
    i = 0
    for result in results:
        res_img[h + vert_offset: 2 * h + vert_offset, i * (w + hor_offset):i * (w + hor_offset) + w] = \
            cv.resize(result.img, display_size)
        i += 1
    split_name = filename.split('.')
    split_name[0] = split_name[0] + "_" + name_addition
    new_filename = split_name[0] + "." + split_name[1]
    cv.imwrite(out_dir_name + "/" + new_filename, res_img)


