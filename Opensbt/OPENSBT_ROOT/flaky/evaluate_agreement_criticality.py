import pandas as pd

# Load data
df = pd.read_csv("./results_flaky/Flakiness_Carla_r-5_n-25/01-02-2026_18-19-37/evaluated_tests.csv")

# Define critical condition
df["is_critical"] = df["Fitness_Min distance"] < 1

# Group by test input (one test = one unique input configuration)
group_cols = ["PedSpeed", "EgoSpeed", "PedDist"]

per_test = (
    df.groupby(group_cols)
      .agg(
          total_runs=("is_critical", "count"),
          critical_runs=("is_critical", "sum")
      )
      .reset_index()
)

# Percentage of critical runs per test
per_test["critical_percentage"] = (
    per_test["critical_runs"] / per_test["total_runs"] * 100
)

# Average percentage across all tests
average_percentage = per_test["critical_percentage"].mean()

print(per_test)
print("Average critical percentage:", average_percentage)