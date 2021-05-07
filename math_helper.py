def make_vector_discrete(vector, discrete_multiplier=1000, discrete_divider=10):
    discrete_vector = (vector * discrete_multiplier) // discrete_divider
    return discrete_vector


def get_manhattan_distance(vector1, vector2, step):
    distance = 0
    # for i in range(self.num_of_colors):
    for i in range(len(vector1)):
    # for i in range(6):
        distance += abs(vector1[i * step] - vector2[i * step])
    return distance
