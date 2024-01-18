# import platform
#
# # 获取处理器信息
# processor_info = platform.processor()
#
# # 打印处理器信息
# print(f"当前处理器：{processor_info}")
#
# # OS: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD
from matplotlib.gridspec import GridSpec
from sko.GA import GA_TSP
from sko.IA import IA_TSP
from sko.SA import SA_TSP
from sko.ACA import ACA_TSP
from sko.PSO import PSO_TSP
import numpy as np
import time
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


if __name__ == '__main__':
    NUM_POINTS = 30
    METHOD = "GA"  # "GA" "IA" "SA" "ACO" "PSO"
    MAX_ITER = 200

    # the map is pre-generated(randomly) in generate.py

    fig = plt.figure(figsize=(20, 4))
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1,1,1])  # 1行2列，宽比例分别为1:1

    plt.rcParams['font.family'] = 'Times New Roman'

    methods = ["GA", "IA", "SA", "ACO", "PSO"]
    # the map is pre-generated(randomly) in generate.py
    for i, METHOD in enumerate(methods):
            points_coordinate = np.load('./results/points_coordinate_{}.npy'.format(NUM_POINTS))
            distance_matrix = np.load('./results/distance_matrix_{}.npy'.format(NUM_POINTS))
            def cal_total_distance(routine):
                num_points, = routine.shape
                return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in
                            range(num_points)])
            start_time = time.time()

            if METHOD == "GA":
                opt_tsp = GA_TSP(func=cal_total_distance, n_dim=NUM_POINTS, size_pop=200, max_iter=MAX_ITER)
            elif METHOD == "IA":
                opt_tsp = IA_TSP(func=cal_total_distance, n_dim=NUM_POINTS, size_pop=200, max_iter=MAX_ITER)
            elif METHOD == "SA":
                opt_tsp = SA_TSP(func=cal_total_distance, x0=range(NUM_POINTS), T_max=100, T_min=1, L=10 * NUM_POINTS)
            elif METHOD == "ACO":
                opt_tsp = ACA_TSP(func=cal_total_distance, n_dim=NUM_POINTS, size_pop=50, max_iter=MAX_ITER,
                                  distance_matrix=distance_matrix)
            elif METHOD == "PSO":
                opt_tsp = PSO_TSP(func=cal_total_distance, n_dim=NUM_POINTS, size_pop=200, max_iter=MAX_ITER, w=0.8,
                                  c1=0.1, c2=0.1)
            else:
                print("wrong method name!")
                opt_tsp = None

            best_points, best_distance = opt_tsp.run()
            print(best_distance, cal_total_distance(best_points))

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"程序运行时间：{elapsed_time:.4f} 秒")

            ax1 = plt.subplot(gs[i])

            best_points_ = np.concatenate([best_points, [best_points[0]]])
            best_points_coordinate = points_coordinate[best_points_, :]

            ax1.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], marker='o', markerfacecolor='lime',
                     color='green', linestyle='-')
            ax1.set_aspect('equal')
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax1.set_xlabel("Longitude")
            ax1.set_ylabel("Latitude")
            plt.subplots_adjust(wspace=0.3)
            ax1.set_title("Optimal Path")

    plt.show()