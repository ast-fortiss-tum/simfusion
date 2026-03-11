import matplotlib.pyplot as plt
import os
from opensbt.utils.duplicates import duplicate_free
import numpy as np
from pathlib import Path


def plot_boxplot(F_all, metrics, save_folder, filename="fitness_boxplot.png"):
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    F_all = np.array(F_all)
    
    if F_all.ndim == 2:
        # Shape: (n_tests, n_repeats)
        n_tests, n_repeats = F_all.shape
        n_objectives = 1
        F_all = F_all[:, :, np.newaxis]  # convert to 3D for uniform handling
    elif F_all.ndim == 3:
        # Shape: (n_tests, n_repeats, n_objectives)
        n_tests, n_repeats, n_objectives = F_all.shape
    else:
        raise ValueError(f"Unexpected F_all shape: {F_all.shape}")

    fig, axs = plt.subplots(1, n_objectives, figsize=(6 * n_objectives, 6), squeeze=False)
    axs = axs[0]  # get the row of axes

    for obj_idx in range(n_objectives):
        ax = axs[obj_idx]
        data = F_all[:, :, obj_idx].T  # shape: (n_repeats, n_tests)
        ax.boxplot(data, patch_artist=True)

        # Set independent y-axis ticks
        y_min, y_max = np.nanmin(data), np.nanmax(data)
        ax.set_ylim(y_min - 0.05 * abs(y_min), y_max + 0.05 * abs(y_max))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Titles and labels
        ax.set_title(f'{n_repeats} Repeats')
        ax.set_xlabel('Test index')
        ax.set_ylabel(f'{metrics[obj_idx]}')

    plt.tight_layout()
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_velocities(V_all, save_folder, filename="velocity_traces.png"):
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    n_repeat = len(V_all)

    # Plotting
    plt.figure(figsize=(10, 6))

    for i, v in enumerate(V_all):
        plt.plot(v, label=f'Run {i+1}', alpha=0.7)

    plt.xlabel('Time step')
    plt.ylabel('Velocity')
    plt.title(f'Velocity Profiles ({n_repeat} Repetitions)')

    # Place legend outside on the right
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        fontsize='small',
        title="Runs"
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend

    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def get_incremented_filename(base_path):
    """
    Returns a non-existing filename by appending _v{n} if needed.
    """
    if not os.path.exists(base_path):
        return base_path
    print("path exists")
    name, ext = os.path.splitext(base_path)
    counter = 1
    while True:
        candidate = f"{name}_v{counter}{ext}"
        print("candidate:", candidate)
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def write_simout_flaky(path, pop, accessor="SO"):
    os.makedirs(path, exist_ok=True)

    for i, _ in enumerate(pop):
        print("len(pop):", len(pop))
        param_values = pop.get("X")[i]
        param_v_chain = "_".join("%.3f" % a for a in param_values)

        simout = pop.get(accessor)[i]
        if simout is None or isinstance(simout, (float, int, np.number)):
            continue

        simout_dumped = simout.to_json()

        filename = f"simout_S{param_v_chain}.json" if param_v_chain is not None else "simout.json"
        full_path = os.path.join(path, filename)

        full_path = get_incremented_filename(full_path)

        with open(full_path, "w") as f:
            f.write(simout_dumped)

def plot_timeseries_from_pop(pop, save_folder, accessor, field, max = 10000, name_folder="trajectories"):
    save_folder_gif = save_folder + os.sep + name_folder
    Path(save_folder_gif).mkdir(parents=True, exist_ok=True)
    
    clean_pop = duplicate_free(pop)[:max]

    if len(clean_pop) == 0:
        return
    simout = clean_pop.get(accessor)[0]
    if simout != None and not (isinstance(simout, (float, int, np.number)) and np.isnan(simout)):
        actors = list((simout.location).keys())
    else:
        return
           
    for index in range(len(clean_pop)):
        param_v_chain = "_".join("%.2f" % a for a in  clean_pop.get("X")[index])

        f = plt.figure(figsize=(12,10))
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, len(actors))]

        simout = clean_pop.get(accessor)[index]
        
        if simout != None and not isinstance(simout, (float, int, np.number)):
            times = simout.times

            param_values = clean_pop.get("X")[index]
            param_v_chain = "_".join("%.2f" % a for a in param_values)
            
            for actor_ind, actor in enumerate(actors):
                if field.lower() == "x":
                    plt.plot(times,
                            [v[0] for v in simout.location[actor]],
                            label=actor,
                            color=colors[actor_ind])
                elif field.lower() == "y":
                    plt.plot(times,
                        [v[1] for v in simout.location[actor]],
                        color=colors[actor_ind],
                        label=actor)
                elif field.lower() == "v":
                    plt.plot(times,
                            [v for v in simout.speed[actor]],
                            color=colors[actor_ind],
                            label=actor)
                elif field.lower() == "a":
                    plt.plot(times,
                            [v for v in simout.acceleration[actor]],
                            color=colors[actor_ind],
                            label=actor)        
                elif field.lower() in simout.otherParams:
                    plt.plot(times,
                        [v for v in simout.otherParams[type.lower()]],
                        color=colors[actor_ind],
                        label=actor)
                else:
                    print("Type is unknown")
                    return
            plt.xlabel('Timestep')
            plt.ylabel(f'{field.upper()}')
            plt.title(f'{field.upper()} Traces')
            plt.legend()
            plt.savefig(save_folder_gif + os.sep + f"{field.upper()}_trace_{param_v_chain}.png", format="png")
            plt.clf()
            plt.close(f)