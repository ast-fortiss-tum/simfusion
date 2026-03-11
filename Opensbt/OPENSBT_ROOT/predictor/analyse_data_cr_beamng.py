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

    # Goal position and velocity parameters
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for cat, color in colors.items():
        subset = df[df['Category'] == cat]
        ax.scatter(subset['goal_center_x'], subset['goal_center_y'], subset['goal_velocity_min'],
                   label=cat, c=color, s=50, edgecolors='k')
    ax.set_title(format_title_with_counts("Design Space (3D) - Goal Parameters", category_counts))
    ax.set_xlabel("Goal Center X")
    ax.set_ylabel("Goal Center Y")
    ax.set_zlabel("Goal Velocity Min")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_folder / f"design_space_3d_goal{name_suffix}.png")
    plt.close()

    # 2D Projections for Goal parameters
    projections = [
        ("goal_center_x", "goal_center_y", f"design_space_goal_x_vs_y{name_suffix}.png", "Goal Position (X vs Y)"),
        ("goal_center_x", "goal_velocity_min", f"design_space_goal_x_vs_vel_min{name_suffix}.png", "Goal X vs Min Velocity"),
        ("goal_center_y", "goal_velocity_min", f"design_space_goal_y_vs_vel_min{name_suffix}.png", "Goal Y vs Min Velocity"),
        ("goal_velocity_min", "goal_velocity_max", f"design_space_vel_min_vs_max{name_suffix}.png", "Velocity Range"),
        ("goal_center_x", "goal_velocity_max", f"design_space_goal_x_vs_vel_max{name_suffix}.png", "Goal X vs Max Velocity"),
        ("goal_center_y", "goal_velocity_max", f"design_space_goal_y_vs_vel_max{name_suffix}.png", "Goal Y vs Max Velocity")
    ]
     
    for x_col, y_col, filename, title in projections:
        fig, ax = plt.subplots(figsize=(8, 6))
        for cat, color in colors.items():
            subset = df[df['Category'] == cat]
            ax.scatter(subset[x_col], subset[y_col], label=cat, c=color, s=50, edgecolors='k')
        ax.set_xlabel(x_col.replace('_', ' ').title())
        ax.set_ylabel(y_col.replace('_', ' ').title())
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

    # LoFi objective plot (filtered) - using obstacle distance and goal distance
    fitness_cols_lofi = [col for col in df.columns if 'Fitness' in col and 'LoFi' in col]
    if len(fitness_cols_lofi) >= 2:
        fitness_x_lofi = fitness_cols_lofi[0]  # First fitness column
        fitness_y_lofi = fitness_cols_lofi[1] if len(fitness_cols_lofi) > 1 else fitness_cols_lofi[0]
        
        # Filter out default values (1000.0)
        df_lofi = df[(df[fitness_x_lofi] != 1000.0) & (df[fitness_y_lofi] != 1000.0)]
        category_counts_lofi = df_lofi['Category'].value_counts().to_dict()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for cat, color in colors.items():
            subset = df_lofi[df_lofi['Category'] == cat]
            ax.scatter(subset[fitness_x_lofi], subset[fitness_y_lofi],
                       label=cat, c=color, s=50, edgecolors='k')
        ax.set_title(format_title_with_counts("Objective Space (LoFi)", category_counts_lofi))
        ax.set_xlabel(fitness_x_lofi.replace('_', ' ').replace('Fitness ', ''))
        ax.set_ylabel(fitness_y_lofi.replace('_', ' ').replace('Fitness ', ''))
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_folder / f"objective_space_lofi_filtered_plot{name_suffix}.png")
        plt.close()

    # HiFi objective plot (filtered)
    fitness_cols_hifi = [col for col in df.columns if 'Fitness' in col and 'HiFi' in col]
    if len(fitness_cols_hifi) >= 2:
        fitness_x_hifi = fitness_cols_hifi[0]  # First fitness column
        fitness_y_hifi = fitness_cols_hifi[1] if len(fitness_cols_hifi) > 1 else fitness_cols_hifi[0]
        
        # Filter out default values (1000.0)
        df_hifi = df[(df[fitness_x_hifi] != 1000.0) & (df[fitness_y_hifi] != 1000.0)]
        category_counts_hifi = df_hifi['Category'].value_counts().to_dict()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for cat, color in colors.items():
            subset = df_hifi[df_hifi['Category'] == cat]
            ax.scatter(subset[fitness_x_hifi], subset[fitness_y_hifi],
                       label=cat, c=color, s=50, edgecolors='k')
        ax.set_title(format_title_with_counts("Objective Space (HiFi)", category_counts_hifi))
        ax.set_xlabel(fitness_x_hifi.replace('_', ' ').replace('Fitness ', ''))
        ax.set_ylabel(fitness_y_hifi.replace('_', ' ').replace('Fitness ', ''))
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_folder / f"objective_space_hifi_filtered_plot{name_suffix}.png")
        plt.close()

def plot_comparison_analysis(input_path: str, save_folder: str, name_suffix: str = ""):
    """Additional analysis specific to CR vs BeamNG comparison"""
    input_path = Path(input_path)
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # Load data
    if input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    else:
        df = pd.read_csv(input_path)
        
    df = df.dropna(subset=['Critical_LoFi', 'Critical_HiFi']) 
    df = classify_criticality(df)

    # Agreement analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Category distribution pie chart
    category_counts = df['Category'].value_counts()
    ax1.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
            colors=[colors[cat] for cat in category_counts.index])
    ax1.set_title("Criticality Agreement Distribution")
    
    # Agreement vs disagreement bar chart
    agreement = category_counts.get('Critical Agreement', 0) + category_counts.get('Non-Critical Agreement', 0)
    disagreement = category_counts.get('LoFi Critical Only', 0) + category_counts.get('HiFi Critical Only', 0)
    
    ax2.bar(['Agreement', 'Disagreement'], [agreement, disagreement], 
            color=['green', 'red'], alpha=0.7)
    ax2.set_title("Overall Agreement vs Disagreement")
    ax2.set_ylabel("Number of Test Cases")
    
    # Add percentage labels
    total = agreement + disagreement
    if total > 0:
        ax2.text(0, agreement/2, f'{agreement/total*100:.1f}%', ha='center', va='center')
        ax2.text(1, disagreement/2, f'{disagreement/total*100:.1f}%', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(save_folder / f"agreement_analysis{name_suffix}.png")
    plt.close()

    # Parameter sensitivity analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    params = ['goal_center_x', 'goal_center_y', 'goal_velocity_min', 'goal_velocity_max']
    
    for i, param in enumerate(params):
        ax = axes[i//2, i%2]
        for cat, color in colors.items():
            subset = df[df['Category'] == cat]
            if len(subset) > 0:
                ax.hist(subset[param], alpha=0.6, label=cat, color=color, bins=10)
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {param.replace("_", " ").title()}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_folder / f"parameter_sensitivity{name_suffix}.png")
    plt.close()

def print_summary_statistics(input_path: str):
    """Print summary statistics about the data"""
    input_path = Path(input_path)
    
    # Load data
    if input_path.is_dir():
        csv_files = list(input_path.glob("*.csv"))
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    else:
        df = pd.read_csv(input_path)
        
    df = df.dropna(subset=['Critical_LoFi', 'Critical_HiFi']) 
    df = classify_criticality(df)
    
    print("\n=== CR vs BeamNG Analysis Summary ===")
    print(f"Total test cases: {len(df)}")
    print(f"\nCriticality Agreement Analysis:")
    
    category_counts = df['Category'].value_counts()
    for cat in colors.keys():
        count = category_counts.get(cat, 0)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    # Overall agreement rate
    agreement = category_counts.get('Critical Agreement', 0) + category_counts.get('Non-Critical Agreement', 0)
    agreement_rate = (agreement / len(df)) * 100 if len(df) > 0 else 0
    print(f"\nOverall Agreement Rate: {agreement_rate:.1f}%")
    
    # Parameter ranges
    print(f"\nParameter Ranges:")
    params = ['goal_center_x', 'goal_center_y', 'goal_velocity_min', 'goal_velocity_max']
    for param in params:
        if param in df.columns:
            print(f"  {param}: {df[param].min():.2f} - {df[param].max():.2f}")

def main():
    input_path = "./results/multisim/"  # Update path for CR BeamNG results
    save_folder = "./predictor/out_cr_beamng/"

    print("Starting CR vs BeamNG analysis...")
    
    # Print summary statistics
    print_summary_statistics(input_path)

    # Generate and save design space plots
    print("Generating design space plots...")
    plot_design_space(input_path, save_folder)

    # Generate and save objective space plots
    print("Generating objective space plots...")
    plot_objective_space(input_path, save_folder)
    
    # Generate comparison analysis plots
    print("Generating comparison analysis plots...")
    plot_comparison_analysis(input_path, save_folder)

    print(f"All plots saved successfully to: {save_folder}")

if __name__ == "__main__":
    main()
