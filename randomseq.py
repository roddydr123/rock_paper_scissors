import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from tqdm import tqdm


def update_grid(grid, grid_size, p1, p2, p3):

    random_probs = np.random.rand(grid_size, grid_size, 3)

    new_grid = np.copy(grid)

    # do a single sweep
    for i in range(grid_size):
        for j in range(grid_size):

            nns = get_nns([i,j], grid, grid_size)

            cell = grid[i,j]

            # if R
            if cell == 0:
                if np.sum(nns == 1) >= 1 and p1 >= random_probs[i,j,0]:
                    new_grid[i,j] = 1
            # if P
            elif cell == 1:
                if np.sum(nns == 2) >= 1 and p2 >= random_probs[i,j,1]:
                    new_grid[i,j] = 2
            # if S
            elif cell == 2:
                if np.sum(nns == 0) >= 1 and p3 >= random_probs[i,j,2]:
                    new_grid[i,j] = 0
            else:
                print("problem")
                sys.exit()

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


def animation(grid, grid_size, p1, p2, p3):

    # if we wanted a visualisation, make a figure for it.
    fig, ax = plt.subplots()
    im = ax.imshow(grid, animated=True)
    cbar = fig.colorbar(im, ax=ax)

    nsteps = 10000

    for n in range(nsteps):

        # move one step forward in the simulation, updating at every point.
        grid = update_grid(grid, grid_size, p1, p2, p3)

        # every 50 sweeps update the animation.
        if n % 1 == 0:
            
            plt.cla()
            im = ax.imshow(grid, interpolation=None, animated=True)
            plt.draw()
            plt.pause(0.00001)


def taskd(grid_size, p1, p2):

    nsteps = 500

    p3_list = np.linspace(0, 0.1, 20)

    N = grid_size**2

    # take data for each value of p3
    for p3 in p3_list:

        grid = np.random.randint(3, size=(grid_size, grid_size))

        minority_fraction_list = []

        # run one simulation
        for n in tqdm(range(nsteps)):

            # move one step forward in the simulation, updating at every point.
            grid = update_grid(grid, grid_size, p1, p2, p3)

            # wait for steady state, 20 looks good enough from animation.
            if n % 1 == 0 and n > 20:
                Rfraction = np.sum(grid == 0) / N
                Pfraction = np.sum(grid == 1) / N
                Sfraction = np.sum(grid == 2) / N

                fractions = np.array([Rfraction, Pfraction, Sfraction])

                # save the smallest fraction from these three.
                minority_fraction_list.append(np.min(fractions))

                # adsorbing state. add zeros for the rest of the simulation.
                if np.any(fractions == 0):
                    minority_fraction_list += [0] * (nsteps - n)
                    break

        np.savetxt(f"data/p3{p3}.dat", minority_fraction_list)


def analyse_taskd():
    p3_list = np.linspace(0, 0.1, 20)
    av = []
    variances = []

    f = open("taskd.dat", "w")
    f.write(f"p3, average minority fraction, variance\n")

    for prob in p3_list:
        ydata = np.loadtxt(f"data/p3{prob}.dat")
        av.append(np.average(ydata))
        variances.append(np.var(ydata))
        f.write(f"{prob}, {np.average(ydata)}, {np.var(ydata)}\n")
    f.close()

    plt.title("Average minority fraction versus p3")
    plt.plot(p3_list, av)
    plt.xlabel("p3")
    plt.ylabel("average minority fraction")
    plt.show()


def taske(grid_size):

    nsteps = 200

    p_space = np.arange(0, 0.3, 0.03)
    p1 = 0.5

    N = grid_size**2

    # take data for each value of p3
    for p3 in p_space:

        for p2 in tqdm(p_space):

            grid = np.random.randint(3, size=(grid_size, grid_size))

            minority_fraction_list = []

            # run one simulation
            for n in range(nsteps):

                # move one step forward in the simulation, updating at every point.
                grid = update_grid(grid, grid_size, p1, p2, p3)

                # wait for steady state, 20 looks good enough from animation.
                if n % 1 == 0 and n > 20:
                    Rfraction = np.sum(grid == 0) / N
                    Pfraction = np.sum(grid == 1) / N
                    Sfraction = np.sum(grid == 2) / N

                    fractions = np.array([Rfraction, Pfraction, Sfraction])

                    # save the smallest fraction from these three.
                    minority_fraction_list.append(np.min(fractions))

                    # adsorbing state. add zeros for the rest of the simulation.
                    if np.any(fractions == 0):
                        minority_fraction_list += [0] * (nsteps - n)
                        break

            np.savetxt(f"data/200-p1-0.5-p2-{p2}-p3-{p3}.dat", minority_fraction_list)


def analyse_taske():
    p_space = np.arange(0, 0.3, 0.03)
    p1 = 0.5

    av = []
    variances = []
    f = open("200-taske.dat", "w")
    f.write(f"p1, p2, p3, average minority fraction, variance\n")

    heatmap = np.zeros((len(p_space), len(p_space)))
    for i, p2 in enumerate(p_space):
        for j, p3 in enumerate(p_space):
            ydata = np.loadtxt(f"data/200-p1-0.5-p2-{p2}-p3-{p3}.dat")
            av.append(np.average(ydata))
            variances.append(np.var(ydata))
            heatmap[i, j] = np.average(ydata)
            f.write(f"{p1}, {p2}, {p3}, {np.average(ydata)}, {np.var(ydata)}\n")
    f.close()

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, origin='lower', extent=(0, 0.3, 0, 0.3))
    cbar = fig.colorbar(im, ax=ax)
    ax.set_title("Average minority fraction for varying p2 and p3")
    ax.set_xlabel("p2")
    ax.set_ylabel("p3")
    plt.show()


def main():
    """Evaluate command line args to choose a function.
    """

    mode = sys.argv[1]

    grid_size = int(sys.argv[2])

    p1 = 0.5
    p2 = 0.5
    p3 = 0.1   # for animation

    if mode == "vis":
        grid = np.random.randint(3, size=(grid_size, grid_size))
        animation(grid, grid_size, p1, p2, p3)
    elif mode == "d":
        taskd(grid_size, p1, p2)
    elif mode == "ad":
        analyse_taskd()
    elif mode == "e":
        taske(grid_size)
    elif mode == "ae":
        analyse_taske()
    else:
        print("wrong args")


main()