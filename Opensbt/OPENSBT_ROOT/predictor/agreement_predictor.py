from pathlib import Path

import numpy as np
import pandas as pd
from glob import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from typing import Dict, Optional, Any, Tuple
import os
import inspect
import json
import matplotlib.pyplot as plt
import seaborn as sns


class AgreementPredictor:
    def __init__(
        self,
        csv_folder: str,
        feature_columns: list,
        k: int = 1,
        model_type: str = "knn",
        model_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_class_weights: bool = True,
        config_path: Optional[str] = None,
        goal_threshold: float = 2.0,
        obstacle_threshold: float = 0.5,
        use_threshold_filtering: bool = True,
    ):
        """
        Initialize the agreement predictor with support for multiple model types.
        
        Parameters:
        - csv_folder: str, path to folder containing CSV files
        - feature_columns: list[str], names of feature columns
        - k: int, number of nearest neighbors for KNN (default = 1)
        - model_type: str, type of model: "knn", "rf", "gboost", "svm", "mlp"
        - model_params: dict, custom model parameters
        - random_state: int, random seed for reproducibility
        - use_class_weights: bool, whether to use balanced class weights
        - config_path: str, path to JSON config file (overrides model_type/model_params)
        - goal_threshold: float, threshold for goal distance (default = 2.0m)
        - obstacle_threshold: float, threshold for obstacle distance (default = 0.5m)
        - use_threshold_filtering: bool, whether to recompute critical labels based on thresholds
        """
        # Load config from JSON if provided
        if config_path is not None:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            model_type = cfg["model_type"]
            model_params = cfg.get("model_params", {})
            goal_threshold = cfg.get("goal_threshold", goal_threshold)
            obstacle_threshold = cfg.get("obstacle_threshold", obstacle_threshold)
            use_threshold_filtering = cfg.get("use_threshold_filtering", use_threshold_filtering)

        self.feature_columns = feature_columns
        self.csv_folder = csv_folder
        self.k = k
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.random_state = random_state
        self.use_class_weights = use_class_weights
        self.goal_threshold = goal_threshold
        self.obstacle_threshold = obstacle_threshold
        self.use_threshold_filtering = use_threshold_filtering

        # Load and concatenate CSV files
        csv_files = glob(f"{csv_folder}/*.csv")
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        # Store original critical labels
        df["Critical_LoFi_Original"] = df["Critical_LoFi"]
        df["Critical_HiFi_Original"] = df["Critical_HiFi"]

        # Recompute critical labels based on thresholds if enabled
        if use_threshold_filtering:
            print(f"\n{'='*80}")
            print("APPLYING THRESHOLD-BASED CRITICAL LABEL FILTERING")
            print(f"{'='*80}")
            print(f"Goal Distance Threshold: {goal_threshold}m")
            print(f"Obstacle Distance Threshold: {obstacle_threshold}m")
            print(f"Criticality Logic: (obstacle < {obstacle_threshold} AND velocity > 0) OR (goal > {goal_threshold})")
            
            # Check if required columns exist
            required_cols = [
                'Fitness_Distance to goal_LoFi',
                'Fitness_Distance to goal_HiFi',
                'Fitness_Min distance to obstacle_LoFi',
                'Fitness_Min distance to obstacle_HiFi',
                'goal_velocity_max'
            ]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for threshold filtering: {missing_cols}")
            
            # Extract distance and velocity columns
            lofi_distance_goal = df['Fitness_Distance to goal_LoFi']
            hifi_distance_goal = df['Fitness_Distance to goal_HiFi']
            lofi_distance_obstacle = df['Fitness_Min distance to obstacle_LoFi']
            hifi_distance_obstacle = df['Fitness_Min distance to obstacle_HiFi']
            velocity = df['goal_velocity_max']
            
            # Recompute criticality
            critical_lofi_new = (
                ((lofi_distance_obstacle < obstacle_threshold) & (velocity > 0)) | 
                (lofi_distance_goal > goal_threshold)
            ).astype(int)
            
            critical_hifi_new = (
                ((hifi_distance_obstacle < obstacle_threshold) & (velocity > 0)) | 
                (hifi_distance_goal > goal_threshold)
            ).astype(int)
            
            # Compare with original labels
            lofi_changed = (critical_lofi_new != df["Critical_LoFi_Original"]).sum()
            hifi_changed = (critical_hifi_new != df["Critical_HiFi_Original"]).sum()
            
            print(f"\nLabel Changes:")
            print(f"  LoFi: {lofi_changed} labels changed ({lofi_changed/len(df)*100:.1f}%)")
            print(f"  HiFi: {hifi_changed} labels changed ({hifi_changed/len(df)*100:.1f}%)")
            
            # Update critical labels
            df["Critical_LoFi"] = critical_lofi_new
            df["Critical_HiFi"] = critical_hifi_new
            
            # Print statistics
            lofi_critical_rate = (critical_lofi_new.sum() / len(df)) * 100
            hifi_critical_rate = (critical_hifi_new.sum() / len(df)) * 100
            
            print(f"\nNew Criticality Rates:")
            print(f"  LoFi: {critical_lofi_new.sum()} critical ({lofi_critical_rate:.1f}%)")
            print(f"  HiFi: {critical_hifi_new.sum()} critical ({hifi_critical_rate:.1f}%)")
            print(f"  Average: {(lofi_critical_rate + hifi_critical_rate)/2:.1f}%")
        else:
            print("\nUsing original critical labels (no threshold filtering)")

        # Derive binary label: 1 = Agreement, 0 = Disagreement
        df["Agreement"] = (df["Critical_LoFi"] == df["Critical_HiFi"]).astype(int)
        
        # Print agreement/disagreement statistics
        agreement_count = df["Agreement"].sum()
        disagreement_count = len(df) - agreement_count
        agreement_rate = (agreement_count / len(df)) * 100
        
        print(f"\nAgreement/Disagreement Distribution:")
        print(f"  Agreement: {agreement_count} cases ({agreement_rate:.1f}%)")
        print(f"  Disagreement: {disagreement_count} cases ({100-agreement_rate:.1f}%)")
        
        both_critical = ((df["Critical_LoFi"] == 1) & (df["Critical_HiFi"] == 1)).sum()
        both_not_critical = ((df["Critical_LoFi"] == 0) & (df["Critical_HiFi"] == 0)).sum()
        lofi_critical_hifi_not = ((df["Critical_LoFi"] == 1) & (df["Critical_HiFi"] == 0)).sum()
        lofi_not_hifi_critical = ((df["Critical_LoFi"] == 0) & (df["Critical_HiFi"] == 1)).sum()
        
        print(f"\nDetailed Breakdown:")
        print(f"  Both Critical: {both_critical}")
        print(f"  Both Not Critical: {both_not_critical}")
        print(f"  LoFi Critical, HiFi Not: {lofi_critical_hifi_not}")
        print(f"  LoFi Not, HiFi Critical: {lofi_not_hifi_critical}")
        print(f"{'='*80}\n")

        print(f"\nDetailed Percentage based Breakdown:")
        print(f"  Both Critical: {both_critical} ({both_critical/len(df)*100:.1f}%)")
        print(f"  Both Not Critical: {both_not_critical} ({both_not_critical/len(df)*100:.1f}%)")
        print(f"  LoFi Critical, HiFi Not: {lofi_critical_hifi_not} ({lofi_critical_hifi_not/len(df)*100:.1f}%)")
        print(f"  LoFi Not, HiFi Critical: {lofi_not_hifi_critical} ({lofi_not_hifi_critical/len(df)*100:.1f}%)")
        print(f"{'='*80}\n")

        # Store the processed dataframe for inspection
        self.df = df

        # Extract features and labels
        self.X = df[feature_columns].to_numpy()
        self.y = df["Agreement"].to_numpy()

        # Compute sample weights (Agreement=1 is minority -> gets higher weights)
        self.sample_weight = (
            compute_sample_weight(class_weight="balanced", y=self.y)
            if self.use_class_weights
            else None
        )

        # Normalize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Fit main model
        self.model = self._make_model()
        self._fit_model(self.model, self.X_scaled, self.y, sample_weight=self.sample_weight)

        # Convenience pipeline (for grid search compatibility)
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("model", self._make_model())])
        self.pipeline.fit(self.X, self.y)

        # Grid search results (filled after running grid_search)
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.grid_search_cv_results_ = None

    def save_processed_data(self, output_path: str):
        """
        Save the processed dataframe with updated critical labels.
        
        Parameters:
        - output_path: str, path to save the CSV file
        """
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")

    def get_threshold_info(self) -> Dict[str, Any]:
        """
        Get information about the thresholds used.
        
        Returns:
        - dict: threshold configuration and statistics
        """
        return {
            'goal_threshold': self.goal_threshold,
            'obstacle_threshold': self.obstacle_threshold,
            'use_threshold_filtering': self.use_threshold_filtering,
            'criticality_logic': f"(obstacle < {self.obstacle_threshold} AND velocity > 0) OR (goal > {self.goal_threshold})",
        }

    def _fit_model(self, estimator: BaseEstimator, X, y, sample_weight=None) -> BaseEstimator:
        """
        Fit estimator, passing sample_weight only if supported.
        Prevents crashes for estimators like KNN (no sample_weight support).
        """
        if sample_weight is None:
            estimator.fit(X, y)
            return estimator

        if "sample_weight" in inspect.signature(estimator.fit).parameters:
            estimator.fit(X, y, sample_weight=sample_weight)
        else:
            estimator.fit(X, y)
        return estimator

    def _make_model(self) -> BaseEstimator:
        """Create model instance based on model_type and model_params."""
        if self.model_type == "knn":
            n_neighbors = self.model_params.get("n_neighbors", self.k)
            params = dict(self.model_params)
            params.pop("n_neighbors", None)
            return KNeighborsClassifier(n_neighbors=n_neighbors, **params)

        if self.model_type == "rf":
            defaults = dict(
                n_estimators=300,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            defaults.update(self.model_params)
            return RandomForestClassifier(**defaults)

        if self.model_type == "gboost":
            defaults = dict(random_state=self.random_state)
            defaults.update(self.model_params)
            return GradientBoostingClassifier(**defaults)

        if self.model_type == "svm":
            defaults = dict(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=self.random_state,
            )
            defaults.update(self.model_params)
            return SVC(**defaults)

        if self.model_type == "mlp":
            defaults = dict(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=3e-4,
                learning_rate_init=5e-4,
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=40,
                random_state=self.random_state,
            )
            defaults.update(self.model_params)
            return MLPClassifier(**defaults)

        raise ValueError(f"Unknown model_type={self.model_type}. Use knn|rf|gboost|svm|mlp")

    def cross_validate(self, cv: int = 5, scoring: Optional[list] = None) -> Dict:
        """
        Perform cross-validation on the dataset.
        
        Parameters:
        - cv: int, number of folds (default = 5)
        - scoring: list[str], scoring metrics
        
        Returns:
        - dict: cross-validation results with mean and std for each metric
        """
        if scoring is None:
            scoring = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_model = self._make_model()

        fit_params = {}
        if self.sample_weight is not None and "sample_weight" in inspect.signature(cv_model.fit).parameters:
            fit_params["sample_weight"] = self.sample_weight

        cv_results = cross_validate(
            cv_model,
            self.X_scaled,
            self.y,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
            fit_params=fit_params if fit_params else None,
        )

        results = {}
        for metric in scoring:
            test_key = f"test_{metric}"
            train_key = f"train_{metric}"
            results[metric] = {
                "test_mean": float(cv_results[test_key].mean()),
                "test_std": float(cv_results[test_key].std()),
                "train_mean": float(cv_results[train_key].mean()),
                "train_std": float(cv_results[train_key].std()),
                "test_scores": cv_results[test_key],
                "train_scores": cv_results[train_key],
            }

        results["fit_time"] = {
            "mean": float(cv_results["fit_time"].mean()),
            "std": float(cv_results["fit_time"].std()),
        }

        return results

    def cross_validate_estimator(
        self,
        estimator: BaseEstimator,
        *,
        cv: int = 5,
        scoring: Optional[list] = None,
    ) -> Dict:
        """
        Cross-validate an arbitrary estimator (e.g., best_estimator_ from GridSearchCV).
        """
        if scoring is None:
            scoring = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        fit_params = {}
        if self.sample_weight is not None:
            if isinstance(estimator, Pipeline) and "model" in estimator.named_steps:
                model_fit_sig = inspect.signature(estimator.named_steps["model"].fit)
                if "sample_weight" in model_fit_sig.parameters:
                    fit_params["model__sample_weight"] = self.sample_weight
            else:
                if "sample_weight" in inspect.signature(estimator.fit).parameters:
                    fit_params["sample_weight"] = self.sample_weight

        cv_results = cross_validate(
            estimator,
            self.X,
            self.y,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
            fit_params=fit_params if fit_params else None,
        )

        results = {}
        for metric in scoring:
            test_key = f"test_{metric}"
            train_key = f"train_{metric}"
            results[metric] = {
                "test_mean": float(cv_results[test_key].mean()),
                "test_std": float(cv_results[test_key].std()),
                "train_mean": float(cv_results[train_key].mean()),
                "train_std": float(cv_results[train_key].std()),
                "test_scores": cv_results[test_key],
                "train_scores": cv_results[train_key],
            }

        results["fit_time"] = {
            "mean": float(cv_results["fit_time"].mean()),
            "std": float(cv_results["fit_time"].std()),
        }

        return results

    def get_validation_certainties(self, cv: int = 5) -> Dict:
        """
        Get certainty distributions for all predictions across validation folds.
        
        Parameters:
        - cv: int, number of folds (default = 5)
        
        Returns:
        - dict: certainty statistics and distributions by fold and class
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_model = self._make_model()
        
        certainties_by_fold = []
        certainties_agreement = []
        certainties_disagreement = []
        all_certainties = []
        all_predictions = []
        all_true_labels = []
        
        fold_details = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_scaled, self.y)):
            X_train, X_val = self.X_scaled[train_idx], self.X_scaled[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # Fit with sample weights if available
            if self.sample_weight is not None and "sample_weight" in inspect.signature(cv_model.fit).parameters:
                sw_train = self.sample_weight[train_idx]
                cv_model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                cv_model.fit(X_train, y_train)
            
            # Get predictions and probabilities
            predictions = cv_model.predict(X_val)
            probabilities = cv_model.predict_proba(X_val)
            
            # Calculate certainties (probability of predicted class)
            fold_certainties = np.array([probabilities[i, predictions[i]] for i in range(len(predictions))])
            
            # Separate by true class
            agreement_mask = y_val == 1
            disagreement_mask = y_val == 0
            
            fold_agreement_certainties = fold_certainties[agreement_mask]
            fold_disagreement_certainties = fold_certainties[disagreement_mask]
            
            certainties_by_fold.append(fold_certainties)
            certainties_agreement.extend(fold_agreement_certainties)
            certainties_disagreement.extend(fold_disagreement_certainties)
            all_certainties.extend(fold_certainties)
            all_predictions.extend(predictions)
            all_true_labels.extend(y_val)
            
            fold_details.append({
                'fold': fold_idx + 1,
                'n_samples': len(y_val),
                'agreement_samples': np.sum(agreement_mask),
                'disagreement_samples': np.sum(disagreement_mask),
                'mean_certainty': np.mean(fold_certainties),
                'std_certainty': np.std(fold_certainties),
                'min_certainty': np.min(fold_certainties),
                'max_certainty': np.max(fold_certainties),
                'median_certainty': np.median(fold_certainties),
                'agreement_mean_certainty': np.mean(fold_agreement_certainties) if len(fold_agreement_certainties) > 0 else 0,
                'disagreement_mean_certainty': np.mean(fold_disagreement_certainties) if len(fold_disagreement_certainties) > 0 else 0,
                'agreement_std_certainty': np.std(fold_agreement_certainties) if len(fold_agreement_certainties) > 0 else 0,
                'disagreement_std_certainty': np.std(fold_disagreement_certainties) if len(fold_disagreement_certainties) > 0 else 0,
            })
        
        # Calculate overall statistics
        all_certainties = np.array(all_certainties)
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        certainties_agreement = np.array(certainties_agreement)
        certainties_disagreement = np.array(certainties_disagreement)
        
        # Correct vs incorrect predictions
        correct_mask = all_predictions == all_true_labels
        certainties_correct = all_certainties[correct_mask]
        certainties_incorrect = all_certainties[~correct_mask]
        
        return {
            'fold_details': fold_details,
            'overall': {
                'mean_certainty': np.mean(all_certainties),
                'std_certainty': np.std(all_certainties),
                'median_certainty': np.median(all_certainties),
                'min_certainty': np.min(all_certainties),
                'max_certainty': np.max(all_certainties),
                'q25_certainty': np.percentile(all_certainties, 25),
                'q75_certainty': np.percentile(all_certainties, 75),
            },
            'by_true_class': {
                'agreement': {
                    'count': len(certainties_agreement),
                    'mean_certainty': np.mean(certainties_agreement),
                    'std_certainty': np.std(certainties_agreement),
                    'median_certainty': np.median(certainties_agreement),
                    'min_certainty': np.min(certainties_agreement),
                    'max_certainty': np.max(certainties_agreement),
                },
                'disagreement': {
                    'count': len(certainties_disagreement),
                    'mean_certainty': np.mean(certainties_disagreement),
                    'std_certainty': np.std(certainties_disagreement),
                    'median_certainty': np.median(certainties_disagreement),
                    'min_certainty': np.min(certainties_disagreement),
                    'max_certainty': np.max(certainties_disagreement),
                }
            },
            'by_correctness': {
                'correct': {
                    'count': len(certainties_correct),
                    'mean_certainty': np.mean(certainties_correct) if len(certainties_correct) > 0 else 0,
                    'std_certainty': np.std(certainties_correct) if len(certainties_correct) > 0 else 0,
                    'median_certainty': np.median(certainties_correct) if len(certainties_correct) > 0 else 0,
                },
                'incorrect': {
                    'count': len(certainties_incorrect),
                    'mean_certainty': np.mean(certainties_incorrect) if len(certainties_incorrect) > 0 else 0,
                    'std_certainty': np.std(certainties_incorrect) if len(certainties_incorrect) > 0 else 0,
                    'median_certainty': np.median(certainties_incorrect) if len(certainties_incorrect) > 0 else 0,
                }
            },
            'raw_data': {
                'all_certainties': all_certainties,
                'certainties_agreement': certainties_agreement,
                'certainties_disagreement': certainties_disagreement,
                'certainties_correct': certainties_correct,
                'certainties_incorrect': certainties_incorrect,
                'predictions': all_predictions,
                'true_labels': all_true_labels,
            }
        }

    def plot_certainty_distributions(self, certainty_data: Dict, save_path: Optional[str] = None):
        """
        Plot certainty distributions from cross-validation.
        
        Parameters:
        - certainty_data: dict, output from get_validation_certainties()
        - save_path: str, optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        threshold_info = f"Goal: {self.goal_threshold}m, Obstacle: {self.obstacle_threshold}m" if self.use_threshold_filtering else "Original Labels"
        fig.suptitle(f'Certainty Distributions - {self.model_type.upper()}\n({threshold_info})', 
                     fontsize=16, fontweight='bold')
        
        raw_data = certainty_data['raw_data']
        
        # 1. Overall certainty distribution
        ax = axes[0, 0]
        ax.hist(raw_data['all_certainties'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(certainty_data['overall']['mean_certainty'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {certainty_data['overall']['mean_certainty']:.3f}")
        ax.axvline(certainty_data['overall']['median_certainty'], color='green', linestyle='--', linewidth=2, 
                   label=f"Median: {certainty_data['overall']['median_certainty']:.3f}")
        ax.set_xlabel('Certainty Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Certainty Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Certainty by true class
        ax = axes[0, 1]
        ax.hist(raw_data['certainties_agreement'], bins=25, alpha=0.6, label='Agreement (True)', color='green', edgecolor='black')
        ax.hist(raw_data['certainties_disagreement'], bins=25, alpha=0.6, label='Disagreement (True)', color='red', edgecolor='black')
        ax.set_xlabel('Certainty Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Certainty Distribution by True Class')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Certainty by correctness
        ax = axes[1, 0]
        ax.hist(raw_data['certainties_correct'], bins=25, alpha=0.6, label='Correct Predictions', color='green', edgecolor='black')
        if len(raw_data['certainties_incorrect']) > 0:
            ax.hist(raw_data['certainties_incorrect'], bins=25, alpha=0.6, label='Incorrect Predictions', color='red', edgecolor='black')
        ax.set_xlabel('Certainty Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Certainty Distribution by Prediction Correctness')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Mean certainty per fold as bar plot
        ax = axes[1, 1]
        fold_numbers = [fold['fold'] for fold in certainty_data['fold_details']]
        fold_means = [fold['mean_certainty'] for fold in certainty_data['fold_details']]
        fold_stds = [fold['std_certainty'] for fold in certainty_data['fold_details']]
        
        bars = ax.bar(fold_numbers, fold_means, yerr=fold_stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axhline(certainty_data['overall']['mean_certainty'], color='red', linestyle='--', linewidth=2, 
                   label=f"Overall Mean: {certainty_data['overall']['mean_certainty']:.3f}")
        ax.set_xlabel('Fold Number')
        ax.set_ylabel('Mean Certainty Score')
        ax.set_title('Mean Certainty per Fold')
        ax.set_ylim([0, 1.0])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
    
    def grid_search(
        self,
        param_grid: Dict[str, list],
        *,
        cv: int = 5,
        refit: str = "f1",
        scoring: Optional[Dict[str, Any]] = None,
        n_jobs: int = -1,
        verbose: int = 1,
        return_train_score: bool = True,
    ) -> Dict[str, Any]:
        """
        Hyperparameter optimization with GridSearchCV on a Pipeline(scaler + model).
        """
        if scoring is None:
            scoring = {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
            }

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", self._make_model()),
            ]
        )

        fit_params = {}
        if self.sample_weight is not None:
            model_fit_sig = inspect.signature(pipe.named_steps["model"].fit)
            if "sample_weight" in model_fit_sig.parameters:
                fit_params["model__sample_weight"] = self.sample_weight

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit,
            cv=skf,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=return_train_score,
        )

        gs.fit(self.X, self.y, **fit_params)

        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_
        self.grid_search_cv_results_ = gs.cv_results_

        return {
            "best_params": gs.best_params_,
            "best_score": float(gs.best_score_),
            "best_estimator": gs.best_estimator_,
            "cv_results": gs.cv_results_,
        }

    def predict(self, x_new: np.ndarray) -> int:
        """Predict agreement/disagreement for a new datapoint."""
        x_new_scaled = self.scaler.transform(x_new)
        return int(self.model.predict(x_new_scaled)[0])

    def predict_with_certainty(self, x_new: np.ndarray) -> Tuple[int, float]:
        """Predict agreement/disagreement with certainty score."""
        x_new_scaled = self.scaler.transform(x_new)
        prediction = int(self.model.predict(x_new_scaled)[0])
        probabilities = self.model.predict_proba(x_new_scaled)[0]
        certainty = probabilities[prediction]
        return prediction, certainty

    def predict_with_label(self, x_new: np.ndarray) -> str:
        """Predict agreement/disagreement with text label."""
        prediction = self.predict(x_new)
        return "Agreement" if prediction == 1 else "Disagreement"

    def predict_with_label_and_certainty(self, x_new: np.ndarray) -> Tuple[str, float]:
        """Predict agreement/disagreement with text label and certainty."""
        prediction, certainty = self.predict_with_certainty(x_new)
        label = "Agreement" if prediction == 1 else "Disagreement"
        return label, certainty

    def get_prediction_details(self, x_new: np.ndarray) -> Dict:
        """
        Get detailed prediction information including probabilities for both classes.
        For KNN, also includes neighbor information.
        """
        x_new_scaled = self.scaler.transform(x_new)
        prediction = int(self.model.predict(x_new_scaled)[0])
        probabilities = self.model.predict_proba(x_new_scaled)[0]
        
        details = {
            'prediction': prediction,
            'label': "Agreement" if prediction == 1 else "Disagreement",
            'certainty': probabilities[prediction],
            'probabilities': {
                'Disagreement': probabilities[0],
                'Agreement': probabilities[1]
            }
        }
        
        # Add neighbor information for KNN models
        if self.model_type == "knn" and hasattr(self.model, 'kneighbors'):
            distances, indices = self.model.kneighbors(x_new_scaled)
            neighbor_labels = self.y[indices[0]]
            
            details['neighbors'] = {
                'k': self.k if hasattr(self, 'k') else self.model.n_neighbors,
                'distances': distances[0],
                'labels': neighbor_labels,
                'agreement_count': np.sum(neighbor_labels == 1),
                'disagreement_count': np.sum(neighbor_labels == 0)
            }
        
        return details

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        unique, counts = np.unique(self.y, return_counts=True)
        distribution = dict(zip(unique, counts))
        return {
            "Disagreement": int(distribution.get(0, 0)),
            "Agreement": int(distribution.get(1, 0)),
            "Total": int(len(self.y)),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_cv_results(results: Dict):
    """Helper function to print cross-validation results in a readable format."""
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    for metric, scores in results.items():
        if metric == 'fit_time':
            print(f"\nFit Time: {scores['mean']:.4f} (±{scores['std']:.4f}) seconds")
        else:
            print(f"\n{metric.upper()}:")
            print(f"  Test:  {scores['test_mean']:.4f} (±{scores['test_std']:.4f})")
            print(f"  Train: {scores['train_mean']:.4f} (±{scores['train_std']:.4f})")
            print(f"  Per-fold test scores: {np.round(scores['test_scores'], 4)}")


def print_certainty_results(certainty_data: Dict):
    """Helper function to print certainty distribution results."""
    print("\n" + "="*80)
    print("VALIDATION CERTAINTY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print("\n--- OVERALL STATISTICS ---")
    overall = certainty_data['overall']
    print(f"Mean Certainty:     {overall['mean_certainty']:.4f}")
    print(f"Std Certainty:      {overall['std_certainty']:.4f}")
    print(f"Median Certainty:   {overall['median_certainty']:.4f}")
    print(f"Min Certainty:      {overall['min_certainty']:.4f}")
    print(f"Max Certainty:      {overall['max_certainty']:.4f}")
    print(f"Q25 Certainty:      {overall['q25_certainty']:.4f}")
    print(f"Q75 Certainty:      {overall['q75_certainty']:.4f}")
    
    # By true class
    print("\n--- BY TRUE CLASS ---")
    agreement_stats = certainty_data['by_true_class']['agreement']
    disagreement_stats = certainty_data['by_true_class']['disagreement']
    
    print(f"\nAgreement (True Class):")
    print(f"  Count:             {agreement_stats['count']}")
    print(f"  Mean Certainty:    {agreement_stats['mean_certainty']:.4f} (±{agreement_stats['std_certainty']:.4f})")
    print(f"  Median Certainty:  {agreement_stats['median_certainty']:.4f}")
    print(f"  Range:             [{agreement_stats['min_certainty']:.4f}, {agreement_stats['max_certainty']:.4f}]")
    
    print(f"\nDisagreement (True Class):")
    print(f"  Count:             {disagreement_stats['count']}")
    print(f"  Mean Certainty:    {disagreement_stats['mean_certainty']:.4f} (±{disagreement_stats['std_certainty']:.4f})")
    print(f"  Median Certainty:  {disagreement_stats['median_certainty']:.4f}")
    print(f"  Range:             [{disagreement_stats['min_certainty']:.4f}, {disagreement_stats['max_certainty']:.4f}]")
    
    # By correctness
    print("\n--- BY PREDICTION CORRECTNESS ---")
    correct_stats = certainty_data['by_correctness']['correct']
    incorrect_stats = certainty_data['by_correctness']['incorrect']
    
    print(f"\nCorrect Predictions:")
    print(f"  Count:             {correct_stats['count']}")
    print(f"  Mean Certainty:    {correct_stats['mean_certainty']:.4f} (±{correct_stats['std_certainty']:.4f})")
    print(f"  Median Certainty:  {correct_stats['median_certainty']:.4f}")
    
    print(f"\nIncorrect Predictions:")
    print(f"  Count:             {incorrect_stats['count']}")
    print(f"  Mean Certainty:    {incorrect_stats['mean_certainty']:.4f} (±{incorrect_stats['std_certainty']:.4f})")
    print(f"  Median Certainty:  {incorrect_stats['median_certainty']:.4f}")
    
    # Per-fold details
    print("\n--- PER-FOLD DETAILS ---")
    for fold in certainty_data['fold_details']:
        print(f"\nFold {fold['fold']} (n={fold['n_samples']}):")
        print(f"  Agreement samples:     {fold['agreement_samples']}")
        print(f"  Disagreement samples:  {fold['disagreement_samples']}")
        print(f"  Mean Certainty:        {fold['mean_certainty']:.4f} (±{fold['std_certainty']:.4f})")
        print(f"  Agreement cert:        {fold['agreement_mean_certainty']:.4f} (±{fold['agreement_std_certainty']:.4f})")
        print(f"  Disagreement cert:     {fold['disagreement_mean_certainty']:.4f} (±{fold['disagreement_std_certainty']:.4f})")


def print_stratified_fold_distribution(y, cv=5):
    """Print class distribution across stratified folds."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    total0 = int((y == 0).sum())
    total1 = int((y == 1).sum())
    total = len(y)

    print("\n" + "="*60)
    print(f"STRATIFIED {cv}-FOLD CLASS DISTRIBUTION")
    print("="*60)
    print(f"Overall: class0={total0} ({total0/total:.2%}), class1={total1} ({total1/total:.2%}), total={total}")

    for i, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        y_val = y[val_idx]
        c0 = int((y_val == 0).sum())
        c1 = int((y_val == 1).sum())
        n = len(y_val)
        print(f"Fold {i}: class0={c0} ({c0/n:.2%}), class1={c1} ({c1/n:.2%}), n={n}")


def write_gridsearch_summary_csv(
    *,
    out_dir: str,
    filename: str,
    batch: str,
    model_type: str,
    refit_metric: str,
    param_grid: Dict[str, Any],
    best_params: Dict[str, Any],
    best_score: float,
    best_cv_results: Dict,
    goal_threshold: float = None,
    obstacle_threshold: float = None,
) -> str:
    """
    One row per (batch, model_type) with best params + best CV metrics (means only).
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    row = {
        "batch": batch,
        "model_type": model_type,
        "refit_metric": refit_metric,
        "best_search_score": float(best_score),
        "searched_param_grid": json.dumps(param_grid, sort_keys=True),
        "best_params": json.dumps(best_params, sort_keys=True),
        "best_cv_accuracy_mean": float(best_cv_results["accuracy"]["test_mean"]),
        "best_cv_balanced_accuracy_mean": float(best_cv_results["balanced_accuracy"]["test_mean"]),
        "best_cv_precision_mean": float(best_cv_results["precision"]["test_mean"]),
        "best_cv_recall_mean": float(best_cv_results["recall"]["test_mean"]),
        "best_cv_f1_mean": float(best_cv_results["f1"]["test_mean"]),
        "goal_threshold": goal_threshold if goal_threshold is not None else "N/A",
        "obstacle_threshold": obstacle_threshold if obstacle_threshold is not None else "N/A",
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, df_row], ignore_index=True)
        combined.to_csv(out_path, index=False)
    else:
        df_row.to_csv(out_path, index=False)

    return out_path


def export_best_model_config(
    *,
    out_dir: str,
    filename: str,
    batch: str,
    model_type: str,
    best_params: Dict[str, Any],
    best_search_score: float,
    best_cv_results: Dict,
    refit_metric: str = "f1",
    goal_threshold: float = 2.0,
    obstacle_threshold: float = 0.5,
    use_threshold_filtering: bool = True,
) -> str:
    """
    Save best model config to JSON so it can be loaded via AgreementPredictor(config_path=...).
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    model_params = {}
    for k, v in best_params.items():
        if k.startswith("model__"):
            model_params[k[len("model__") :]] = v

    payload = {
        "batch": batch,
        "model_type": model_type,
        "model_params": model_params,
        "refit_metric": refit_metric,
        "best_search_score": float(best_search_score),
        "goal_threshold": goal_threshold,
        "obstacle_threshold": obstacle_threshold,
        "use_threshold_filtering": use_threshold_filtering,
        "best_cv_means": {
            "accuracy": float(best_cv_results["accuracy"]["test_mean"]),
            "balanced_accuracy": float(best_cv_results["balanced_accuracy"]["test_mean"]),
            "precision": float(best_cv_results["precision"]["test_mean"]),
            "recall": float(best_cv_results["recall"]["test_mean"]),
            "f1": float(best_cv_results["f1"]["test_mean"]),
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return out_path


def pareto_front(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Extract Pareto-optimal rows from dataframe based on specified metrics."""
    if df.empty:
        return df

    values = df[metrics].to_numpy()
    n = len(df)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        ge = (values >= values[i]).all(axis=1)
        gt = (values > values[i]).any(axis=1)
        dominators = ge & gt
        dominators[i] = False
        if dominators.any():
            dominated[i] = True

    return df.loc[~dominated].copy()


def collect_all_gridsearch_summaries(*, results_root: str = "./results", filename: str = "gridsearch_summary.csv") -> pd.DataFrame:
    """Collect all grid search summaries from subdirectories."""
    csv_paths = glob(os.path.join(results_root, "*", filename))
    frames = []
    for p in csv_paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def report_best_models_pareto(
    *,
    results_root: str = "./results",
    filename: str = "gridsearch_summary.csv",
    metrics: Optional[list] = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """Report Pareto-optimal models across all batches and model types."""
    if metrics is None:
        metrics = ["best_cv_f1_mean", "best_cv_recall_mean"]

    df = collect_all_gridsearch_summaries(results_root=results_root, filename=filename)
    if df.empty:
        print("No grid search summaries found yet.")
        return df

    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    df = df.dropna(subset=metrics)

    pf = pareto_front(df, metrics=metrics)
    pf_sorted = pf.sort_values(by=metrics, ascending=[False] * len(metrics)).head(top_k)

    cols_to_show = ["batch", "model_type"] + metrics + ["best_search_score", "best_params", "goal_threshold", "obstacle_threshold"]
    cols_to_show = [c for c in cols_to_show if c in pf_sorted.columns]

    print("\n" + "=" * 80)
    print("PARETO-OPTIMAL MODELS (across all batches + models)")
    print("Maximizing: " + ", ".join(metrics))
    print("=" * 80)
    print(pf_sorted[cols_to_show].to_string(index=False))

    return pf_sorted


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Training and evaluation workflow with threshold filtering
    batches = ["data_final"]
    run_grid_search = True
    grid_refit_metric = "f1"
    gridsearch_summary_csv = "gridsearch_summary.csv"
    
    # Threshold configuration
    GOAL_THRESHOLD = 2.0  # meters - adjust this value
    OBSTACLE_THRESHOLD = 1  # meters
    USE_THRESHOLD_FILTERING = True  # Set to False to use original labels

    for batch_path in batches:
        csv_folder = f"./predictor/data_cr_beamng/{batch_path}/"
        features=[
        "goal_center_x",
        "goal_center_y",
        "goal_velocity_min",
        "goal_velocity_max",
        "num_npc",
        "lane_count",
        "npc_x0",
        "npc_v_max",
        "npc_seed",
        "cutin_tx",
        "cutin_delay",
        ]
        param_grids = {
            "knn": {
                "model__n_neighbors": [3, 5],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "rf": {
                "model__n_estimators": [300, 800],
                "model__max_depth": [None, 10, 25],
                "model__min_samples_split": [2, 5],
            },
            "gboost": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__max_depth": [2, 3],
            },
            "svm": {
                "model__C": [0.5, 1.0, 3.0],
                "model__gamma": ["scale", 0.1, 0.01],
                "model__kernel": ["rbf"],
            },
            "mlp": {
                "model__hidden_layer_sizes": [(64, 32), (128, 64, 32)],
                "model__alpha": [1e-4, 3e-4, 1e-3],
                "model__learning_rate_init": [1e-3, 5e-4],
            },
        }

        model_configs = [
            ("knn", {"n_neighbors": 3}),
            ("rf", {"n_estimators": 1000, "max_depth": None}),
            ("gboost", {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3}),
            ("svm", {"C": 3.0, "gamma": "scale", "kernel": "rbf", "probability": True}),
            ("mlp", {"hidden_layer_sizes": (128, 64, 32), "alpha": 3e-4, "max_iter": 1500}),
        ]

        metrics_out_dir = f"./results_th_obs1/{batch_path}/"
        Path(metrics_out_dir).mkdir(parents=True, exist_ok=True)
        for model_type, model_params in model_configs:
            print("\n" + "#" * 80)
            print(f"MODEL: {model_type.upper()}   initial_params={model_params}")
            print("#" * 80)

            predictor = AgreementPredictor(
                csv_folder=csv_folder,
                feature_columns=features,
                k=3,
                model_type=model_type,
                model_params=model_params,
                use_class_weights=True,
                goal_threshold=GOAL_THRESHOLD,
                obstacle_threshold=OBSTACLE_THRESHOLD,
                use_threshold_filtering=USE_THRESHOLD_FILTERING,
            )

            # Optionally save the processed data with new labels
            # predictor.save_processed_data(f"{metrics_out_dir}processed_data_{model_type}.csv")

            dist = predictor.get_class_distribution()
            print(
                f"Class distribution: 0={dist['Disagreement']}, 1={dist['Agreement']} "
                f"(Agreement={dist['Agreement']/dist['Total']:.2%})"
            )

            # Perform cross-validation
            cv_results = predictor.cross_validate(cv=5)
            print_cv_results(cv_results)

            # Get certainty distributions
            certainty_data = predictor.get_validation_certainties(cv=5)
            print_certainty_results(certainty_data)

            # Plot certainty distributions
            predictor.plot_certainty_distributions(
                certainty_data, 
                save_path=f"{metrics_out_dir}certainty_{model_type}_goal{GOAL_THRESHOLD}_obs{OBSTACLE_THRESHOLD}.png"
            )

            # Grid search
            if run_grid_search and model_type in param_grids:
                gs_out = predictor.grid_search(
                    param_grid=param_grids[model_type],
                    cv=5,
                    refit=grid_refit_metric,
                    n_jobs=-1,
                    verbose=1,
                )
                print(f"Best params: {gs_out['best_params']}")
                print(f"Best search {grid_refit_metric}: {gs_out['best_score']:.4f}")

                best_cv_results = predictor.cross_validate_estimator(
                    gs_out["best_estimator"],
                    cv=5,
                    scoring=["accuracy", "balanced_accuracy", "precision", "recall", "f1"],
                )
                print_cv_results(best_cv_results)

                csv_path = write_gridsearch_summary_csv(
                    out_dir=metrics_out_dir,
                    filename=gridsearch_summary_csv,
                    batch=batch_path,
                    model_type=model_type,
                    refit_metric=grid_refit_metric,
                    param_grid=param_grids[model_type],
                    best_params=gs_out["best_params"],
                    best_score=gs_out["best_score"],
                    best_cv_results=best_cv_results,
                    goal_threshold=GOAL_THRESHOLD,
                    obstacle_threshold=OBSTACLE_THRESHOLD,
                )
                print(f"Wrote/updated grid search summary CSV: {csv_path}")

                cfg_path = export_best_model_config(
                    out_dir=metrics_out_dir,
                    filename=f"best_model_config_{model_type}_goal{GOAL_THRESHOLD}_obs{OBSTACLE_THRESHOLD}.json",
                    batch=batch_path,
                    model_type=model_type,
                    best_params=gs_out["best_params"],
                    best_search_score=gs_out["best_score"],
                    best_cv_results=best_cv_results,
                    refit_metric=grid_refit_metric,
                    goal_threshold=GOAL_THRESHOLD,
                    obstacle_threshold=OBSTACLE_THRESHOLD,
                    use_threshold_filtering=USE_THRESHOLD_FILTERING,
                )
                print(f"Exported best model config: {cfg_path}")

        report_best_models_pareto(
            results_root="./results",
            filename=gridsearch_summary_csv,
            metrics=["best_cv_f1_mean", "best_cv_recall_mean"],
            top_k=25,
        )