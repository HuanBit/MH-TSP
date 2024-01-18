import numpy as np
from scipy import spatial

# 1. generate and save 3 sets of map, each set includes 5 maps (10,20,30,40,50)
# 2. calculate the corresponding distance_matrix and save

if __name__ == '__main__':
    num_points = 50
    for num_points in range(10, 60, 10):
        points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
        distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
        np.save('./results/points_coordinate_{}_2.npy'.format(num_points), points_coordinate)
        np.save('./results/distance_matrix_{}_2.npy'.format(num_points), distance_matrix)

    # points_coordinate = np.load('./results/points_coordinate_{}.npy'.format(num_points))
    # distance_matrix = np.load('./results/distance_matrix_{}.npy'.format(num_points))
