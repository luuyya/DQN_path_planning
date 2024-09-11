import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_random_map(size=100):

    map = np.zeros((size, size))

    obstacle_num = int(size * size * 0.1)
    np.random.seed(34)
    obstacle_coords = np.random.randint(0, size, size=(obstacle_num, 2))

    for coord in obstacle_coords:
        map[coord[0], coord[1]] = 1  # 用1表示障碍物

    # plt.figure(figsize=(10, 10))
    # plt.imshow(map, cmap="gray", origin="upper")
    # plt.show()

    print(type(map))

    return map

def initialize(map):
    size,_=np.shape(map)
    np.random.seed(34)
    tmp=np.random.randint(low=0,high=size,size=(2,2))
    start=tmp[0]
    end=tmp[1]

    while map[start[0],start[1]]==0 and map[end[0],end[1]]==0:
        tmp = np.random.randint((2, 2))
        start = tmp[0]
        end = tmp[1]

    return start,end


def plot_map(map, start, end):
    colored_map = np.copy(map)

    fig, ax = plt.subplots()

    colored_map[start[0], start[1]] = 2
    colored_map[end[0], end[1]] = 3

    new_cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])

    ax.imshow(colored_map, cmap=new_cmap)
    plt.show()

map=create_random_map()
start,end=initialize(map)
print(start,end)
plot_map(map, start, end)

