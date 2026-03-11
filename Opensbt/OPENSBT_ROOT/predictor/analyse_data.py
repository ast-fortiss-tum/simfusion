import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define colors for each category
colors = {
    'Critical Agreement': 'red',
    'Non-Critical Agreement': 'green',
    'LoFi Critical Only': 'blue',
    'HiFi Critical Only': 'orange'
}

def classify_criticality(df: pd.DataFrame) -> pd.DataFrame:
    def get_category(row):
        if row['Critical_LoFi'] == 1 and row['Critical_HiFi'] == 1:
            return 'Critical Agreement'
        elif row['Critical_LoFi'] == 0 and row['Critical_HiFi'] == 0:
            return 'Non-Critical Agreement'
        elif row['Critical_LoFi'] == 1:
            return 'LoFi Critical Only'
        else:
            return 'HiFi Critical Only'
    df['Category'] = df.apply(get_category, axis=1)
    return df

def format_title_with_counts(base_title: str, category_counts: dict) -> str:
    keys = list(colors.keys())
    line1 = ", ".join([f"{cat}: {category_counts.get(cat, 0)}" for cat in keys[:2]])
    line2 = ", ".join([f"{cat}: {category_counts.get(cat, 0)}" for cat in keys[2:]])
    return f"{base_title}\n{line1}\n{line2}"

def plot_design_space(input_path: str, save_folder: str, name_suffix: str = ""):
    input_path = Path(input_path)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # Load and merge CSVs if folder, or load single file
    if input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    else:
        df = pd.read_csv(input_path)

    df = df.dropna(subset=['Critical_LoFi', 'Critical_HiFi']) 

    df = classify_criticality(df)
    category_counts = df['Category'].value_counts().to_dict()

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cat, color in colors.items():
        subset = df[df['Category'] == cat]
        ax.scatter(subset['PedSpeed'], subset['EgoSpeed'], subset['PedDist'],
                   label=cat, c=color, s=50, edgecolors='k')
    ax.set_title(format_title_with_counts("Design Space (3D)", category_counts))
    ax.set_xlabel("PedSpeed")
    ax.set_ylabel("EgoSpeed")
    ax.set_zlabel("PedDist")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_folder / "design_space_3d.png")
    plt.close()

    # 2D Projections
    projections = [
        ("PedSpeed", "EgoSpeed", f"design_space_Ped_vs_Ego{name_suffix}.png", "Top-Down View"),
        ("PedSpeed", "PedDist", f"design_space_Ped_vs_Dist{name_suffix}.png", "Side View 1"),
        ("EgoSpeed", "PedDist", f"design_space_Ego_vs_Dist{name_suffix}.png", "Side View 2")
    ]
     
    for x_col, y_col, filename, title in projections:
        fig, ax = plt.subplots(figsize=(8, 6))
        for cat, color in colors.items():
            subset = df[df['Category'] == cat]
            ax.scatter(subset[x_col], subset[y_col], label=cat, c=color, s=50, edgecolors='k')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(format_title_with_counts(title, category_counts))
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_folder / filename)
        plt.close()

def plot_objective_space(input_path: str, save_folder: str, name_suffix: str = ""):
    input_path = Path(input_path)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # Load and merge CSVs if folder, or load single file
    if input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    else:
        df = pd.read_csv(input_path)
        
    df = df.dropna(subset=['Critical_LoFi', 'Critical_HiFi']) 

    df = classify_criticality(df)

    # LoFi objective plot (filtered)
    df_lofi = df[(df['Fitness_Min distance_LoFi'] != 1000.0) & (df['Fitness_Velocity at min distance_LoFi'] != 1000.0)]
    category_counts_lofi = df_lofi['Category'].value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(8, 6))
    for cat, color in colors.items():
        subset = df_lofi[df_lofi['Category'] == cat]
        ax.scatter(subset['Fitness_Min distance_LoFi'], 
                   subset['Fitness_Velocity at min distance_LoFi'],
                   label=cat, c=color, s=50, edgecolors='k')
    ax.set_title(format_title_with_counts("Objective Space (LoFi)", category_counts_lofi))
    ax.set_xlabel("Fitness_Min distance_LoFi")
    ax.set_ylabel("Fitness_Velocity at min distance_LoFi")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_folder / f"objective_space_lofi_filtered_plot{name_suffix}.png")
    plt.close()

    # HiFi objective plot (filtered)
    df_hifi = df[(df['Fitness_Min distance_HiFi'] != 1000.0) & (df['Fitness_Velocity at min distance_HiFi'] != 1000.0)]
    category_counts_hifi = df_hifi['Category'].value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(8, 6))
    for cat, color in colors.items():
        subset = df_hifi[df_hifi['Category'] == cat]
        ax.scatter(subset['Fitness_Min distance_HiFi'], subset['Fitness_Velocity at min distance_HiFi'],
                   label=cat, c=color, s=50, edgecolors='k')
    ax.set_title(format_title_with_counts("Objective Space (HiFi)", category_counts_hifi))
    ax.set_xlabel("Fitness_Min distance_HiFi")
    ax.set_ylabel("Fitness_Velocity at min distance_HiFi")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_folder / f"objective_space_hifi_filtered_plot{name_suffix}.png")
    plt.close()

def main():
    input_path = "./predictor/data/"
    save_folder = "./predictor/out/"

    # Generate and save design space plots
    plot_design_space(input_path, save_folder)

    # Generate and save objective space plots
    plot_objective_space(input_path, save_folder)

    print(f"Plots saved successfully to: {save_folder}")

if __name__ == "__main__":
    main()