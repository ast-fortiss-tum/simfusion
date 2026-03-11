from predictor.agreement_predictor import AgreementPredictor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple
import pandas as pd


class ThresholdOptimizer:
    """
    Optimize classification threshold based on business requirements.
    """
    
    def __init__(self, predictor: AgreementPredictor, cv: int = 5):
        """
        Initialize threshold optimizer.
        
        Parameters:
        - predictor: trained AgreementPredictor instance
        - cv: number of cross-validation folds
        """
        self.predictor = predictor
        self.cv = cv
        self.threshold_results = None
        self.optimal_thresholds = None
        
    def get_cv_probabilities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predicted probabilities and true labels across CV folds.
        
        Returns:
        - y_true: true labels
        - y_proba: predicted probabilities for positive class
        """
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        cv_model = self.predictor._make_model()
        
        all_proba = []
        all_true = []
        
        for train_idx, val_idx in skf.split(self.predictor.X_scaled, self.predictor.y):
            X_train, X_val = self.predictor.X_scaled[train_idx], self.predictor.X_scaled[val_idx]
            y_train, y_val = self.predictor.y[train_idx], self.predictor.y[val_idx]
            
            # Fit with sample weights if available
            if (self.predictor.sample_weight is not None and 
                "sample_weight" in cv_model.fit.__code__.co_varnames):
                sw_train = self.predictor.sample_weight[train_idx]
                cv_model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                cv_model.fit(X_train, y_train)
            
            # Get probabilities for positive class (Agreement = 1)
            proba = cv_model.predict_proba(X_val)[:, 1]
            
            all_proba.extend(proba)
            all_true.extend(y_val)
        
        return np.array(all_true), np.array(all_proba)
    
    def evaluate_threshold(self, y_true: np.ndarray, y_proba: np.ndarray, 
                          threshold: float) -> Dict:
        """
        Evaluate metrics at a specific threshold.
        
        Parameters:
        - y_true: true labels
        - y_proba: predicted probabilities
        - threshold: classification threshold
        
        Returns:
        - dict of evaluation metrics
        """
        y_pred = (y_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate (Miss Rate)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2
        
        # Geometric mean
        g_mean = np.sqrt(recall * specificity)
        
        # Matthews Correlation Coefficient
        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'g_mean': g_mean,
            'mcc': mcc,
            'fpr': fpr,
            'fnr': fnr,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
        }
    
    def find_optimal_thresholds(self, y_true: np.ndarray, y_proba: np.ndarray,
                               strategies: List[str] = None) -> Dict:
        """
        Find optimal thresholds using different strategies.
        
        Parameters:
        - y_true: true labels
        - y_proba: predicted probabilities
        - strategies: list of optimization strategies
        
        Returns:
        - dict mapping strategy names to optimal thresholds and metrics
        """
        if strategies is None:
            strategies = [
                'f1',
                'balanced_accuracy', 
                'g_mean',
                'youden',
                'cost_sensitive',
                'high_recall',
                'high_precision'
            ]
        
        optimal = {}
        
        # Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_proba)
        
        # ROC curve
        fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_proba)
        
        # F1 maximization
        if 'f1' in strategies:
            f1_scores = 2 * (precision_curve[:-1] * recall_curve[:-1]) / (
                precision_curve[:-1] + recall_curve[:-1] + 1e-10
            )
            best_idx = np.argmax(f1_scores)
            optimal['f1'] = self.evaluate_threshold(y_true, y_proba, pr_thresholds[best_idx])
            optimal['f1']['strategy'] = 'Maximize F1 Score'
        
        # Balanced Accuracy maximization
        if 'balanced_accuracy' in strategies:
            thresholds = np.linspace(0, 1, 100)
            balanced_accs = []
            for t in thresholds:
                metrics = self.evaluate_threshold(y_true, y_proba, t)
                balanced_accs.append(metrics['balanced_accuracy'])
            best_idx = np.argmax(balanced_accs)
            optimal['balanced_accuracy'] = self.evaluate_threshold(
                y_true, y_proba, thresholds[best_idx]
            )
            optimal['balanced_accuracy']['strategy'] = 'Maximize Balanced Accuracy'
        
        # G-Mean maximization (good for imbalanced datasets)
        if 'g_mean' in strategies:
            g_means = np.sqrt(tpr_curve * (1 - fpr_curve))
            best_idx = np.argmax(g_means)
            optimal['g_mean'] = self.evaluate_threshold(y_true, y_proba, roc_thresholds[best_idx])
            optimal['g_mean']['strategy'] = 'Maximize G-Mean (Imbalanced Datasets)'
        
        # Youden's J statistic (maximize TPR - FPR)
        if 'youden' in strategies:
            j_scores = tpr_curve - fpr_curve
            best_idx = np.argmax(j_scores)
            optimal['youden'] = self.evaluate_threshold(y_true, y_proba, roc_thresholds[best_idx])
            optimal['youden']['strategy'] = "Youden's J Statistic (Maximize TPR - FPR)"
        
        # Cost-sensitive (assuming FN is more costly than FP)
        if 'cost_sensitive' in strategies:
            # Cost ratio: False Negative is 2x more costly than False Positive
            fn_cost = 2.0
            fp_cost = 1.0
            
            thresholds = np.linspace(0, 1, 100)
            costs = []
            for t in thresholds:
                metrics = self.evaluate_threshold(y_true, y_proba, t)
                cost = (metrics['fn'] * fn_cost + metrics['fp'] * fp_cost)
                costs.append(cost)
            best_idx = np.argmin(costs)
            optimal['cost_sensitive'] = self.evaluate_threshold(
                y_true, y_proba, thresholds[best_idx]
            )
            optimal['cost_sensitive']['strategy'] = 'Cost-Sensitive (FN=2x FP)'
        
        # High Recall (minimize false negatives)
        if 'high_recall' in strategies:
            # Find threshold that achieves at least 90% recall with best precision
            target_recall = 0.90
            valid_thresholds = pr_thresholds[recall_curve[:-1] >= target_recall]
            valid_precisions = precision_curve[:-1][recall_curve[:-1] >= target_recall]
            
            if len(valid_thresholds) > 0:
                best_idx = np.argmax(valid_precisions)
                optimal['high_recall'] = self.evaluate_threshold(
                    y_true, y_proba, valid_thresholds[best_idx]
                )
                optimal['high_recall']['strategy'] = f'High Recall (≥{target_recall:.0%})'
            else:
                # If can't achieve target recall, use threshold that maximizes recall
                best_idx = np.argmax(recall_curve[:-1])
                optimal['high_recall'] = self.evaluate_threshold(
                    y_true, y_proba, pr_thresholds[best_idx]
                )
                optimal['high_recall']['strategy'] = 'Maximize Recall'
        
        # High Precision (minimize false positives)
        if 'high_precision' in strategies:
            # Find threshold that achieves at least 90% precision with best recall
            target_precision = 0.90
            valid_thresholds = pr_thresholds[precision_curve[:-1] >= target_precision]
            valid_recalls = recall_curve[:-1][precision_curve[:-1] >= target_precision]
            
            if len(valid_thresholds) > 0:
                best_idx = np.argmax(valid_recalls)
                optimal['high_precision'] = self.evaluate_threshold(
                    y_true, y_proba, valid_thresholds[best_idx]
                )
                optimal['high_precision']['strategy'] = f'High Precision (≥{target_precision:.0%})'
            else:
                # If can't achieve target precision, use threshold that maximizes precision
                best_idx = np.argmax(precision_curve[:-1])
                optimal['high_precision'] = self.evaluate_threshold(
                    y_true, y_proba, pr_thresholds[best_idx]
                )
                optimal['high_precision']['strategy'] = 'Maximize Precision'
        
        return optimal
    
    def analyze_thresholds(self, n_thresholds: int = 100) -> pd.DataFrame:
        """
        Analyze performance across a range of thresholds.
        
        Parameters:
        - n_thresholds: number of thresholds to evaluate
        
        Returns:
        - DataFrame with metrics for each threshold
        """
        y_true, y_proba = self.get_cv_probabilities()
        
        thresholds = np.linspace(0, 1, n_thresholds)
        results = []
        
        for t in thresholds:
            metrics = self.evaluate_threshold(y_true, y_proba, t)
            results.append(metrics)
        
        self.threshold_results = pd.DataFrame(results)
        return self.threshold_results
    
    def optimize(self, strategies: List[str] = None) -> Dict:
        """
        Run full threshold optimization pipeline.
        
        Parameters:
        - strategies: list of optimization strategies
        
        Returns:
        - dict of optimal thresholds per strategy
        """
        print("Getting cross-validation probabilities...")
        y_true, y_proba = self.get_cv_probabilities()
        
        print("Finding optimal thresholds...")
        self.optimal_thresholds = self.find_optimal_thresholds(y_true, y_proba, strategies)
        
        print("Analyzing threshold range...")
        self.analyze_thresholds()
        
        return self.optimal_thresholds
    
    def plot_threshold_analysis(self, save_path: str = None):
        """
        Plot comprehensive threshold analysis.
        
        Parameters:
        - save_path: optional path to save the figure
        """
        if self.threshold_results is None:
            raise ValueError("Run analyze_thresholds() first")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Threshold Analysis', fontsize=16, fontweight='bold')
        
        df = self.threshold_results
        
        # 1. Precision-Recall vs Threshold
        ax = axes[0, 0]
        ax.plot(df['threshold'], df['precision'], label='Precision', linewidth=2)
        ax.plot(df['threshold'], df['recall'], label='Recall', linewidth=2)
        ax.plot(df['threshold'], df['f1'], label='F1 Score', linewidth=2, linestyle='--')
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default (0.5)')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision-Recall-F1 vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Accuracy metrics vs Threshold
        ax = axes[0, 1]
        ax.plot(df['threshold'], df['accuracy'], label='Accuracy', linewidth=2)
        ax.plot(df['threshold'], df['balanced_accuracy'], label='Balanced Accuracy', linewidth=2)
        ax.plot(df['threshold'], df['g_mean'], label='G-Mean', linewidth=2, linestyle='--')
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Accuracy Metrics vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Confusion matrix components
        ax = axes[0, 2]
        total = df['tp'] + df['fp'] + df['tn'] + df['fn']
        ax.plot(df['threshold'], df['tp'] / total, label='TP Rate', linewidth=2)
        ax.plot(df['threshold'], df['fp'] / total, label='FP Rate', linewidth=2)
        ax.plot(df['threshold'], df['tn'] / total, label='TN Rate', linewidth=2)
        ax.plot(df['threshold'], df['fn'] / total, label='FN Rate', linewidth=2)
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('Confusion Matrix Rates vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Recall vs Specificity (Trade-off)
        ax = axes[1, 0]
        ax.plot(df['threshold'], df['recall'], label='Recall (Sensitivity)', linewidth=2)
        ax.plot(df['threshold'], df['specificity'], label='Specificity', linewidth=2)
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('Sensitivity-Specificity Trade-off')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 5. Error rates
        ax = axes[1, 1]
        ax.plot(df['threshold'], df['fpr'], label='False Positive Rate', linewidth=2, color='red')
        ax.plot(df['threshold'], df['fnr'], label='False Negative Rate', linewidth=2, color='orange')
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title('Error Rates vs Threshold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 6. Optimal thresholds visualization
        ax = axes[1, 2]
        if self.optimal_thresholds:
            strategies = list(self.optimal_thresholds.keys())
            thresholds_opt = [self.optimal_thresholds[s]['threshold'] for s in strategies]
            f1_scores = [self.optimal_thresholds[s]['f1'] for s in strategies]
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
            bars = ax.barh(strategies, thresholds_opt, color=colors, alpha=0.7)
            ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default (0.5)')
            ax.set_xlabel('Optimal Threshold')
            ax.set_title('Optimal Thresholds by Strategy')
            ax.set_xlim([0, 1])
            ax.legend()
            ax.grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_roc_and_pr_curves(self, save_path: str = None):
        """
        Plot ROC and Precision-Recall curves with optimal thresholds marked.
        
        Parameters:
        - save_path: optional path to save the figure
        """
        y_true, y_proba = self.get_cv_probabilities()
        
        # Calculate curves
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('ROC and Precision-Recall Curves', fontsize=16, fontweight='bold')
        
        # ROC Curve
        ax = axes[0]
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        # Mark optimal thresholds
        if self.optimal_thresholds:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimal_thresholds)))
            for (strategy, metrics), color in zip(self.optimal_thresholds.items(), colors):
                # Find closest point on ROC curve
                threshold = metrics['threshold']
                idx = np.argmin(np.abs(roc_thresholds - threshold))
                ax.plot(fpr[idx], tpr[idx], 'o', markersize=8, color=color, 
                       label=f"{strategy} (t={threshold:.3f})")
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Recall)')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        # Precision-Recall Curve
        ax = axes[1]
        ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        
        # Baseline (random classifier for imbalanced data)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(baseline, color='k', linestyle='--', linewidth=1, 
                  label=f'Random Classifier (p={baseline:.3f})')
        
        # Mark optimal thresholds
        if self.optimal_thresholds:
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.optimal_thresholds)))
            for (strategy, metrics), color in zip(self.optimal_thresholds.items(), colors):
                recall_val = metrics['recall']
                precision_val = metrics['precision']
                ax.plot(recall_val, precision_val, 'o', markersize=8, color=color,
                       label=f"{strategy} (t={metrics['threshold']:.3f})")
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def print_optimal_thresholds(self):
        """Print optimal thresholds with detailed metrics."""
        if not self.optimal_thresholds:
            raise ValueError("Run optimize() first")
        
        print("\n" + "=" * 90)
        print("OPTIMAL THRESHOLDS BY STRATEGY")
        print("=" * 90)
        
        for strategy, metrics in self.optimal_thresholds.items():
            print(f"\n{metrics['strategy'].upper()}")
            print("-" * 90)
            print(f"Threshold:          {metrics['threshold']:.4f}")
            print(f"Accuracy:           {metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
            print(f"Precision:          {metrics['precision']:.4f}")
            print(f"Recall:             {metrics['recall']:.4f}")
            print(f"Specificity:        {metrics['specificity']:.4f}")
            print(f"F1 Score:           {metrics['f1']:.4f}")
            print(f"G-Mean:             {metrics['g_mean']:.4f}")
            print(f"MCC:                {metrics['mcc']:.4f}")
            print(f"FPR:                {metrics['fpr']:.4f}")
            print(f"FNR:                {metrics['fnr']:.4f}")
            print(f"Confusion Matrix:   TP={metrics['tp']}, FP={metrics['fp']}, "
                  f"TN={metrics['tn']}, FN={metrics['fn']}")


def recommend_threshold(optimizer: ThresholdOptimizer, use_case: str = "balanced") -> float:
    """
    Recommend a threshold based on the use case.
    
    Parameters:
    - optimizer: ThresholdOptimizer with optimized results
    - use_case: one of "balanced", "safety_critical", "cost_effective", "high_confidence"
    
    Returns:
    - recommended threshold value
    """
    if not optimizer.optimal_thresholds:
        raise ValueError("Run optimize() first")
    
    recommendations = {
        "balanced": {
            "strategy": "f1",
            "description": "Balanced trade-off between precision and recall"
        },
        "safety_critical": {
            "strategy": "high_recall",
            "description": "Minimize false negatives (catch all agreements)"
        },
        "cost_effective": {
            "strategy": "cost_sensitive",
            "description": "Minimize total cost (considering FN > FP)"
        },
        "high_confidence": {
            "strategy": "high_precision",
            "description": "Minimize false positives (only predict when confident)"
        },
        "imbalanced": {
            "strategy": "g_mean",
            "description": "Good for imbalanced datasets (geometric mean of sensitivity/specificity)"
        },
    }
    
    if use_case not in recommendations:
        raise ValueError(f"Unknown use case. Choose from: {list(recommendations.keys())}")
    
    rec = recommendations[use_case]
    strategy = rec["strategy"]
    threshold = optimizer.optimal_thresholds[strategy]["threshold"]
    
    print("\n" + "=" * 70)
    print(f"RECOMMENDATION: {use_case.upper()}")
    print("=" * 70)
    print(f"Strategy: {rec['description']}")
    print(f"Recommended Threshold: {threshold:.4f}")
    print("\nMetrics at this threshold:")
    
    metrics = optimizer.optimal_thresholds[strategy]
    for key in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1']:
        print(f"  {key:20s}: {metrics[key]:.4f}")
    
    return threshold


if __name__ == "__main__":
    # Load model
    from predictor.agreement_predictor import AgreementPredictor
    
    config_path = "./predictor/models/cr/best_model_config_rf_goal2.0_obs0.5.json"
    
    goal_threshold = 2
    obstacle_threshold = 1

    suffix = "_cr"

    print("Loading model...")
    predictor = AgreementPredictor(
        csv_folder="./predictor/data_cr_beamng/data_final/",
        feature_columns=[
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
        ],
        config_path=config_path,
        use_threshold_filtering=True,
        goal_threshold=goal_threshold,
        obstacle_threshold=obstacle_threshold
    )
    
    print(f"Model type: {predictor.model_type.upper()}")
    print(f"Class distribution: {predictor.get_class_distribution()}")
    
    # Initialize optimizer
    optimizer = ThresholdOptimizer(predictor, cv=5)
    
    # Run optimization
    optimal_thresholds = optimizer.optimize()
    
    # Print results
    optimizer.print_optimal_thresholds()
    
    # Visualize
    optimizer.plot_threshold_analysis(save_path=f"./predictor/analysis/threshold_analysis_gth{goal_threshold}_oth{obstacle_threshold}{suffix}.png")
    optimizer.plot_roc_and_pr_curves(save_path=f"./predictor/analysis/roc_pr_curves_gth{goal_threshold}_oth{obstacle_threshold}{suffix}.png")
    
    # Get recommendations for different use cases
    print("\n\n" + "#" * 90)
    print("THRESHOLD RECOMMENDATIONS FOR DIFFERENT USE CASES")
    print("#" * 90)
    
    for use_case in ["balanced", "safety_critical", "cost_effective", "high_confidence", "imbalanced"]:
        threshold = recommend_threshold(optimizer, use_case)
        print()