class Image:

    def __init__(self, file_name, dir_name, img, vector, discrete_vector):
        self.file_name = file_name
        self.dir_name = dir_name
        self.img = img
        self.vector = vector
        self.discrete_vector = discrete_vector

    def get_ith_discrete_mean(self, i):
        return self.discrete_vector[i * 2]

    def get_ith_discrete_std_dev(self, i):
        return self.discrete_vector[i * 2 + 1]
