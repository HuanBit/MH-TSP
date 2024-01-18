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
    NUM_POINTS = 20
    MAX_ITER = 200

    # the map is pre-generated(randomly) in generate.py
    for METHOD in ["ACO"]:  # ["GA", "IA", "SA", "ACO", "PSO"]:
        for NUM_POINTS in range(10, 60, 10):
            points_coordinate = np.load('./results/points_coordinate_{}.npy'.format(NUM_POINTS))
            distance_matrix = np.load('./results/distance_matrix_{}.npy'.format(NUM_POINTS))

            def cal_total_distance(routine):
                num_points, = routine.shape
                return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

            start_time = time.time()

            # 5method
            if METHOD == "GA":  # 1. genetic algorithm
                opt_tsp = GA_TSP(func=cal_total_distance, n_dim=NUM_POINTS, size_pop=200, max_iter=MAX_ITER)
            elif METHOD == "IA":  # 2. Immune algorithm
                opt_tsp = IA_TSP(func=cal_total_distance, n_dim=NUM_POINTS, size_pop=200, max_iter=MAX_ITER)
            elif METHOD == "SA":  # 3. Simulated Annealing
                opt_tsp = SA_TSP(func=cal_total_distance, x0=range(NUM_POINTS), T_max=100, T_min=1, L=10 * NUM_POINTS)
            elif METHOD == "ACO":  # 4. Ant Colony Optimization
                opt_tsp = ACA_TSP(func=cal_total_distance, n_dim=NUM_POINTS, size_pop=50, max_iter=MAX_ITER, distance_matrix=distance_matrix)
            elif METHOD == "PSO":  # 5. Particle Swarm Optimization
                opt_tsp = PSO_TSP(func=cal_total_distance, n_dim=NUM_POINTS, size_pop=200, max_iter=MAX_ITER, w=0.8, c1=0.1, c2=0.1)
            else:
                print("wrong method name!")
                opt_tsp = None

            best_points, best_distance = opt_tsp.run()
            print(best_distance, cal_total_distance(best_points))

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"程序运行时间：{elapsed_time:.4f} 秒")

            plt.rcParams['font.family'] = 'Times New Roman'
            fig = plt.figure(figsize=(10,5))
            gs = GridSpec(1, 2, width_ratios=[1,1])  # 1行2列，高度比例分别为1:1
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])

            best_points_ = np.concatenate([best_points, [best_points[0]]])
            best_points_coordinate = points_coordinate[best_points_, :]

            if METHOD == "PSO":
                ax0.plot(opt_tsp.gbest_y_hist, color='green')
            else:
                ax0.plot(opt_tsp.generation_best_Y, color='green')
            # ax[0].set_aspect('equal')
            ax0.set_xlabel("Iteration")
            ax0.set_ylabel("Distance")
            ax0.set_title("Shortest Distance")

            ax1.plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], marker='o', markerfacecolor='lime', color='green', linestyle='-')
            ax1.set_aspect('equal')
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax1.set_xlabel("Longitude")
            ax1.set_ylabel("Latitude")
            plt.subplots_adjust(wspace=0.3)
            ax1.set_title("Optimal Path")
            # plt.show()

            plt.savefig(f'./results/figs/{METHOD}_{NUM_POINTS}_{MAX_ITER}_1')
            plt.close(fig)
            with open('results/figs/results.txt', 'a') as file:
                file.write(f"{METHOD}_{NUM_POINTS}_{MAX_ITER}_1 \n")
                file.write(f"程序运行时间：{elapsed_time:.4f} \n")
                file.write(f"最短距离：{cal_total_distance(best_points)} \n")


