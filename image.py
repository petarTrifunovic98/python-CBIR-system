import json


class Image:

    def __init__(self, file_name, dir_name, img, vector, discrete_vector, hist_vector, tex_vector):
        img_config_file = open('./img_config.json')
        self.img_config = json.load(img_config_file)
        self.file_name = file_name
        self.dir_name = dir_name
        self.img = img
        self.vector = vector
        self.discrete_vector = discrete_vector
        self.hist_vector = hist_vector
        self.tex_vector = tex_vector

    def get_ith_discrete_mean(self, i):
        return self.discrete_vector[i * 2]

    def get_ith_discrete_std_dev(self, i):
        return self.discrete_vector[i * 2 + 1]

    def get_ith_discrete_glcm_prop(self, i):
        num_glcm_props = len(self.img_config['glcm_props'])
        return self.discrete_vector[-num_glcm_props + i]
