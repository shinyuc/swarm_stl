import pickle
from vis import vis

with open('plot_low_level_env.pkl', 'rb') as file:
    PWLs, Boxes_output, plots = pickle.load(file)

vis(plots, PWLs, data_width=6.4, data_height = 6.4)