import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_mae_over_time(csv_path: str, save_plot: bool = True) -> None:
    """
    Loads MAE data from a CSV file, processes it, and plots the MAE over time.

    Parameters:
    - csv_path (str): Path to the CSV file containing MAE data.
    - save_plot (bool): Whether to save the plot as a PNG file in the same directory.
    """
    # Load and process data
    df = pd.read_csv(csv_path)
    df['mae'] = df['mae'].apply(lambda x: list(map(float, x.strip('[]').split())))
    df['mae1'] = df['mae'].apply(lambda x: x[0])
    df['mae2'] = df['mae'].apply(lambda x: x[1])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['mae1'], label='Distance MAE', color='blue')
    plt.plot(df['step'], df['mae2'], label='Velocity MAE', color='orange')
    plt.xlabel('Scenario Evaluation')
    plt.ylabel('MAE')
    plt.title('Velocity/Distance MAE Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save or show plot
    if save_plot:
        output_path = os.path.join(os.path.dirname(csv_path), "mae_plot.png")
        plt.savefig(output_path, format="png")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    path_mae = "./results/Surrogate_RF_Testing_seed-310_pop-20_gen-20_time-3h/NSGA-II/01-07-2025_01-29-17/model_out/mae.csv"
    plot_mae_over_time(path_mae)
