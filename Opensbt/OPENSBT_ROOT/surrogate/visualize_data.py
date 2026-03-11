import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_data(X: np.ndarray, 
                   y_class: np.ndarray, 
                   output_folder: str, 
                   filename: str = "data_plot.png",
                   interactive: bool = False):
    """
    Visualizes critical vs. non-critical data in a 3D scatter plot.

    Parameters:
        X (np.ndarray): Input features of shape (n_samples, 3). Assumes columns are [PedSpeed, EgoSpeed, PedDist].
        y_class (np.ndarray): Binary labels indicating critical (1) or non-critical (0) for each sample.
        output_folder (str): Directory to save the plot.
        filename (str): Filename for the saved plot image (default: "data_plot.png").
        interactive (bool): If True, display the plot interactively. Otherwise, only save the plot.
    """
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, filename)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    num_critical = np.sum(y_class == 1)
    num_total = len(y_class)
    
    # Plot non-critical points
    mask_non_critical = (y_class == 0)
    ax.scatter(X[mask_non_critical, 0], X[mask_non_critical, 1], X[mask_non_critical, 2],
               c='blue', label='Non-critical', alpha=0.5)

    # Plot critical points
    mask_critical = (y_class == 1)
    ax.scatter(X[mask_critical, 0], X[mask_critical, 1], X[mask_critical, 2],
               c='red', label='Critical', alpha=0.7)

    ax.set_xlabel('PedSpeed')
    ax.set_ylabel('EgoSpeed')
    ax.set_zlabel('PedDist')
    ax.set_title(f'Critical/Non-critical Test Scenarios ({num_critical}/{num_total - num_critical})')
    ax.legend()

    if interactive:
        plt.show()
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_data(X: np.ndarray, 
                   y_class: np.ndarray, 
                   output_folder: str, 
                   filename: str = "data_plot.png",
                   interactive: bool = False):
    """
    Visualizes critical vs. non-critical data in a 3D scatter plot.

    Parameters:
        X (np.ndarray): Input features of shape (n_samples, 3). Assumes columns are [PedSpeed, EgoSpeed, PedDist].
        y_class (np.ndarray): Binary labels indicating critical (1) or non-critical (0) for each sample.
        output_folder (str): Directory to save the plot.
        filename (str): Filename for the saved plot image (default: "data_plot.png").
        interactive (bool): If True, display the plot interactively. Otherwise, only save the plot.
    """
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, filename)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    num_critical = np.sum(y_class == 1)
    num_total = len(y_class)
    
    # Plot non-critical points
    mask_non_critical = (y_class == 0)
    ax.scatter(X[mask_non_critical, 0], X[mask_non_critical, 1], X[mask_non_critical, 2],
               c='blue', label='Non-critical', alpha=0.5)

    # Plot critical points
    mask_critical = (y_class == 1)
    ax.scatter(X[mask_critical, 0], X[mask_critical, 1], X[mask_critical, 2],
               c='red', label='Critical', alpha=0.7)

    ax.set_xlabel('PedSpeed')
    ax.set_ylabel('EgoSpeed')
    ax.set_zlabel('PedDist')
    ax.set_title(f'Critical/Non-critical Test Scenarios ({num_critical}/{num_total - num_critical})')
    ax.legend()

    if interactive:
        plt.show()
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    from glob import glob
    import pandas as pd
    from simulations.config import critical_function

    input_columns = ["PedSpeed", "EgoSpeed", "PedDist"]
    optional_columns =  ["Fitness_Min distance", "Fitness_Velocity at min distance"]
    desired_columns = ["Fitness_Min distance_HiFi", "Fitness_Velocity at min distance_HiFi"]
    data_folder = "./surrogate/data/batch5/"
    model_output_path = "./surrogate/out/model_"
    output_folder = "./surrogate/plots/"
    save_folder = "./surrogate/out/"

    csv_files = glob(f"{data_folder}/*.csv")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    target_columns = []

    if optional_columns[0] in df.columns:
        target_columns = optional_columns
    else:
        target_columns = desired_columns

    df = df[input_columns + target_columns].dropna()

    # Remove rows with dummy values
    for col in target_columns:
        df = df[df[col] != 1000].reset_index(drop=True)

    ######### get criticality
    print("num tests (training):", len(df))
    df_fit = df[target_columns]

    # Extract tuples of fitness values
    fitness_tuples = list(df_fit.itertuples(index=False, name=None))
    label_values = [critical_function.eval(np.asarray(t)) for t in fitness_tuples]
    critical_tests = [t for t in fitness_tuples if critical_function.eval(np.asarray(t))]
    print("num critical: ", len(critical_tests))
    print("non critical tests: ", len(df_fit) - len(critical_tests))

    #########################
    X = df[input_columns].values
    y_reg = df[target_columns].values
    y_class = np.asarray(label_values)

    visualize_data(X, y_class, output_folder=output_folder, interactive=True)