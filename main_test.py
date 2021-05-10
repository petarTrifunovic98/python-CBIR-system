from image_processor import ImageProcessor
from image_retrieval_service import ImageRetrievalService
from sorting_strategies import RedisSortStrategy
from image_repository import ImageRepository

image_processor = ImageProcessor(3, [1], [0])
sorting_strategy = RedisSortStrategy('localhost', 6379)
image_repository = ImageRepository('./images')

service = ImageRetrievalService('./images', image_processor, sorting_strategy, image_repository)
#service.add_images('../images')

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
                print(results)

print("Goodbye!")
