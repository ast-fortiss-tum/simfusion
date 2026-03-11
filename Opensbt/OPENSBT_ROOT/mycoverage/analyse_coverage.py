from mycoverage.compute_coverage import coverage_pipeline
from mycoverage.filter import match_critical_simouts

if __name__ == "__main__":
    
    # -------------------------
    # Approach 1 (LoFi)
    # -------------------------
    approach1_folder = "/home/user/testing/topic1-opensbt-aw/results_wandb/LoFi_GA_pop10_t06_00_00_seed10/oqbwxfkz"
    approach1_csv = f"{approach1_folder}/all_critical_testcases.csv"
    approach1_simout = f"{approach1_folder}/simout"
    approach1_critical = match_critical_simouts(approach1_csv, approach1_simout)
    
    # -------------------------
    # Approach 2
    # -------------------------
    approach2_folder = "/home/user/testing/topic1-opensbt-aw/results_wandb/HiFi_GA_pop10_t06_00_00_seed10/1mypwuxq"
    approach2_csv = f"{approach2_folder}/all_critical_testcases.csv"
    approach2_simout = f"{approach2_folder}/simout"
    approach2_critical = match_critical_simouts(approach2_csv, approach2_simout)
    
    # -------------------------
    # Approach 3
    # -------------------------
    approach3_folder = "/home/user/testing/topic1-opensbt-aw/results_wandb/Predict_GA_pop10_t06_00_00_k3_seed10/fc0ivh5o"
    approach3_csv = f"{approach3_folder}/all_critical_testcases.csv"
    approach3_simout = f"{approach3_folder}/simout_hifi"
    approach3_critical = match_critical_simouts(approach3_csv, approach3_simout)
    
    # -------------------------
    # List of critical simouts per approach
    # -------------------------
    sources = [
        approach1_critical, 
        approach2_critical,
        approach3_critical
    ]
    
    # -------------------------
    # Run full pipeline: clustering, t-SNE, coverage, entropy
    # -------------------------
    vectors_std, labels, simout_mapping, cluster_to_scenarios, simout_paths, coverage_dict, entropy_dict = coverage_pipeline(
        sources,
        eps=1.5,
        min_samples=2
    )
    
    # -------------------------
    # Print coverage and entropy
    # -------------------------
    print("\n[RESULT] Coverage per approach:")
    for app, cov in coverage_dict.items():
        print(f"Approach {app+1}: Coverage = {cov:.2f}")
        
    print("\n[RESULT] Normalized Entropy per approach:")
    for app, ent in entropy_dict.items():
        print(f"Approach {app+1}: Entropy = {ent:.2f}")