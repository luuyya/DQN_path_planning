from utils.env import Map
from utils.plot import plot_map

def main():
    map_env = Map(8, 0.1, 34)

    grid = map_env.create_random_map()

    start, end = map_env.initialize_start_end()

    plot_map(grid, start, end)

if __name__ == "__main__":
    main()