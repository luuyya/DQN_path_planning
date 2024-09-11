import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import utils.env as env


def plot_map(map, start, end):
    colored_map = np.copy(map)

    fig, ax = plt.subplots()

    colored_map[start[0], start[1]] = 2
    colored_map[end[0], end[1]] = 3

    new_cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])

    ax.imshow(colored_map, cmap=new_cmap)
    plt.show()

map = env.map.create_random_map(10)
start,end = env.map.initialize(map)
print(start,end)
plot_map(map, start, end)

