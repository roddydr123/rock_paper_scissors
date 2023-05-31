import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from tqdm import tqdm


def update_grid(grid, grid_size):

    new_grid = np.copy(grid)

    # do a single sweep
    for i in range(grid_size):
        for j in range(grid_size):

            nns = get_nns([i,j], grid, grid_size)

            cell = grid[i,j]

            # if R
            if cell == 0:
                if np.sum(nns == 1) >= 3:
                    new_grid[i,j] = 1
            # if P
            elif cell == 1:
                if np.sum(nns == 2) > 2:
                    new_grid[i,j] = 2
            # if S
            elif cell == 2:
                if np.sum(nns == 0) > 2:
                    new_grid[i,j] = 0
            else:
                print("problem")

    grid = new_grid

    return grid


def get_nns(indices, grid, grid_size):

    nn_spins = [
        grid[(indices[0] + 1)%grid_size, indices[1]],
        grid[indices[0] - 1, indices[1]],
        grid[indices[0], indices[1] - 1],
        grid[indices[0], (indices[1] + 1)%grid_size],
        grid[(indices[0] + 1)%grid_size, (indices[1] + 1)%grid_size],
        grid[indices[0] - 1, (indices[1] + 1)%grid_size],
        grid[(indices[0] + 1)%grid_size, (indices[1] - 1)],
        grid[(indices[0] - 1), (indices[1] - 1)]
        ]
    return np.array(nn_spins)


def pie_grid(grid_size):
    grid = np.zeros((grid_size, grid_size))

    # left edge, j =0
    # right edge, j = grid_size
    # top, i = 0
    for i in range(grid_size):
        for j in range(grid_size):
            d_to_left = abs(j - 0)
            d_to_right = abs(j - grid_size)
            d_to_top = abs(i - 0)
            choices = [d_to_left, d_to_right, d_to_top]
            closest = np.argmin(choices)

            # if close to left, assign to R
            if closest == 0:
                grid[i,j] = 0
            # closest to right assign P
            elif closest == 1:
                grid[i,j] = 1
            elif closest == 2:
                grid[i,j] = 2

    return grid


def animation(grid, grid_size):

    fig, ax = plt.subplots()
    im = ax.imshow(grid, animated=True)
    cbar = fig.colorbar(im, ax=ax)

    nsteps = 10000

    for n in range(nsteps):

        # move one step forward in the simulation, updating at every point.
        grid = update_grid(grid, grid_size)

        if n % 5 == 0:
            
            plt.cla()
            im = ax.imshow(grid, interpolation=None, animated=True)
            plt.draw()
            plt.pause(0.00001)


def taskb(grid, grid_size):

    nsteps = 1000
    total_Rs = []
    steps = []

    for n in tqdm(range(nsteps)):

        # move one step forward in the simulation, updating at every point.
        grid = update_grid(grid, grid_size)

        if n % 1 == 0:
            
            total_Rs.append(np.sum(grid == 0))
            steps.append(n)
    
    np.savetxt("taskb.dat", total_Rs)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(steps, total_Rs)
    ax1.set_title("Total number of R states in grid over time")
    ax1.set_ylabel("# of R states")
    ax1.set_xlabel("# steps")
    plt.show()


def main():
    """Evaluate command line args to choose a function.
    """

    mode = sys.argv[1]

    grid_size = int(sys.argv[2])

    if mode == "vis":
        grid = pie_grid(grid_size)
        animation(grid, grid_size)
    elif mode == "b":
        grid = pie_grid(grid_size)
        taskb(grid, grid_size)
    else:
        print("wrong args")


main()