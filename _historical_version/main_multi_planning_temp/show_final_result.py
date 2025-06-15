from main_planning import task_environment
import matplotlib.pyplot as plt; plt.rcdefaults()
import plot_figures as pf
import pickle
import numpy as np
from bounding_box_plot import plot_ellipse_2d

_, plots = task_environment()
plt.rcParams["figure.figsize"] = [6.4, 6.4]
plt.rcParams['axes.titlesize'] = 20
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('off')

vertices = []
for plot in plots:
    for A, b in plot[0]:
        pf.plot_polygon_ge(A, b, color=plot[1])

plt.gca().set_aspect('equal', adjustable='box')

with open('PWLs_output_time.pkl', 'rb') as file:
    PWLs, margin_loaded, Box_loaded = pickle.load(file)
if len(PWLs) <= 4:
    colors = ['k', np.array([153, 0, 71]) / 255, np.array([6, 0, 153]) / 255, np.array([0, 150, 0]) / 255]
else:
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in np.linspace(0, 0.85, len(PWLs))]

for i in range(len(PWLs)):
    PWL = PWLs[i]
    ax.plot([P[0][0] for P in PWL], [P[0][1] for P in PWL], '-', color=colors[i])
    ax.plot(PWL[-1][0][0], PWL[-1][0][1], '*', color=colors[i])
    ax.plot(PWL[0][0][0], PWL[0][0][1], 'o', color=colors[i])
    for j in range(len(PWL)):
        x0 = np.array(PWL[j][0])
        box = Box_loaded[j]
        plot_ellipse_2d(x0, box)

plt.show()