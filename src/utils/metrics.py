"""
Metrics calculation utilities for VACE

Provides comprehensive metrics calculation including:
- Top-1 and Top-5 accuracy
- Per-class metrics (F1, Recall, Precision)
- Calibration metrics (ECE, NLL, Brier score)
- Margin statistics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
import torch


class MetricsCalculator:
    """Calculate various metrics for classification tasks"""
    
    def __init__(self):
        pass
    
    def calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
        class_names: Optional[List[str]] = None,
        num_bins: int = 15
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics
        
        Args:
            predictions: Predicted class indices (N,)
            targets: True class indices (N,)
            probabilities: Class probabilities (N, num_classes)
            class_names: List of class names
            num_bins: Number of bins for ECE calculation
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic accuracy metrics
        metrics.update(self._calculate_accuracy_metrics(predictions, targets, probabilities))
        
        # Per-class metrics
        metrics.update(self._calculate_per_class_metrics(predictions, targets))
        
        # Calibration metrics
        metrics.update(self._calculate_calibration_metrics(predictions, targets, probabilities, num_bins))
        
        # Confusion matrix (as numpy array for logging)
        metrics['confusion_matrix'] = confusion_matrix(targets, predictions)
        
        return metrics
    
    def _calculate_accuracy_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """Calculate accuracy-based metrics"""
        metrics = {}
        
        # Top-1 accuracy
        metrics['top1_accuracy'] = 100.0 * np.mean(predictions == targets)
        
        # Top-5 accuracy (if applicable)
        num_classes = probabilities.shape[1]
        if num_classes >= 5:
            top5_pred = np.argsort(probabilities, axis=1)[:, -5:]
            top5_correct = np.any(top5_pred == targets.reshape(-1, 1), axis=1)
            metrics['top5_accuracy'] = 100.0 * np.mean(top5_correct)
        
        return metrics
    
    def _calculate_per_class_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate per-class metrics"""
        metrics = {}
        
        # Macro-averaged metrics (unweighted average across classes)
        metrics['macro_f1'] = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(targets, predictions, average='macro', zero_division=0)
        metrics['macro_precision'] = precision_score(targets, predictions, average='macro', zero_division=0)
        
        # Weighted metrics (weighted by class frequency)
        metrics['weighted_f1'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(targets, predictions, average='weighted', zero_division=0)
        metrics['weighted_precision'] = precision_score(targets, predictions, average='weighted', zero_division=0)
        
        return metrics
    
    def _calculate_calibration_metrics(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        probabilities: np.ndarray, 
        num_bins: int = 15
    ) -> Dict[str, float]:
        """Calculate calibration metrics"""
        metrics = {}
        
        # Get max probabilities (confidence)
        max_probs = np.max(probabilities, axis=1)
        correct = (predictions == targets).astype(float)
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(max_probs, correct, num_bins)
        metrics['ece'] = ece
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(max_probs, correct, num_bins)
        metrics['mce'] = mce
        
        # Negative Log-Likelihood (NLL)
        # Get probabilities of true classes
        true_class_probs = probabilities[np.arange(len(targets)), targets]
        nll = -np.mean(np.log(np.clip(true_class_probs, 1e-12, 1.0)))
        metrics['nll'] = nll
        
        # Brier Score (multi-class version)
        brier_score = self._calculate_brier_score(targets, probabilities)
        metrics['brier_score'] = brier_score
        
        return metrics
    
    def _calculate_ece(self, confidences: np.ndarray, correct: np.ndarray, num_bins: int) -> float:
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if sample is in bin m (between bin lower and upper)
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, confidences: np.ndarray, correct: np.ndarray, num_bins: int) -> float:
        """Calculate Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _calculate_brier_score(self, targets: np.ndarray, probabilities: np.ndarray) -> float:
        """Calculate multi-class Brier Score"""
        num_classes = probabilities.shape[1]
        
        # Convert targets to one-hot encoding
        targets_onehot = np.eye(num_classes)[targets]
        
        # Brier score = mean squared difference between predicted and true probabilities
        brier_score = np.mean(np.sum((probabilities - targets_onehot) ** 2, axis=1))
        
        return brier_score
    
    def calculate_margin_statistics(
        self, 
        logits: np.ndarray, 
        targets: np.ndarray, 
        tau: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate margin statistics for analysis
        
        Args:
            logits: Model logits (N, num_classes)
            targets: True targets (N,)
            tau: Temperature values (num_classes,) - for VACE analysis
            
        Returns:
            Dictionary of margin statistics
        """
        if tau is not None:
            # Scale logits by temperature for VACE
            scaled_logits = logits / tau.reshape(1, -1)
        else:
            scaled_logits = logits
        
        # Calculate margins: (correct_logit - max_incorrect_logit) / tau_correct
        margins = []
        
        for i in range(len(targets)):
            target_class = targets[i]
            correct_logit = scaled_logits[i, target_class]
            
            # Get maximum logit from incorrect classes
            incorrect_mask = np.ones(scaled_logits.shape[1], dtype=bool)
            incorrect_mask[target_class] = False
            max_incorrect = np.max(scaled_logits[i][incorrect_mask])
            
            margin = correct_logit - max_incorrect
            margins.append(margin)
        
        margins = np.array(margins)
        
        return {
            'margin_mean': np.mean(margins),
            'margin_std': np.std(margins),
            'margin_min': np.min(margins),
            'margin_max': np.max(margins),
            'margin_median': np.median(margins),
            'negative_margin_ratio': np.mean(margins < 0)  # Ratio of negative margins
        }
    
    def get_reliability_diagram_data(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        probabilities: np.ndarray, 
        num_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for plotting reliability diagram
        
        Returns:
            Tuple of (bin_confidences, bin_accuracies, bin_counts)
        """
        max_probs = np.max(probabilities, axis=1)
        correct = (predictions == targets).astype(float)
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                count_in_bin = in_bin.sum()
                
                bin_confidences.append(avg_confidence_in_bin)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(count_in_bin)
            else:
                bin_confidences.append(0.0)
                bin_accuracies.append(0.0)
                bin_counts.append(0)
        
        return np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts)
    
    def calculate_per_class_accuracy(
        self, 
        predictions: np.ndarray, 
        targets: np.ndarray, 
        num_classes: int
    ) -> np.ndarray:
        """Calculate per-class accuracy"""
        per_class_acc = np.zeros(num_classes)
        
        for c in range(num_classes):
            mask = targets == c
            if mask.sum() > 0:
                per_class_acc[c] = (predictions[mask] == targets[mask]).mean()
            else:
                per_class_acc[c] = 0.0  # No samples for this class
        
        return per_class_acc


def format_metrics_for_logging(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary for nice logging output"""
    lines = []
    
    if prefix:
        lines.append(f"=== {prefix} Metrics ===")
    
    # Group metrics by type
    accuracy_metrics = {k: v for k, v in metrics.items() if 'accuracy' in k}
    calibration_metrics = {k: v for k, v in metrics.items() if k in ['ece', 'mce', 'nll', 'brier_score']}
    per_class_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['macro', 'weighted'])}
    margin_metrics = {k: v for k, v in metrics.items() if 'margin' in k}
    
    # Format each group
    if accuracy_metrics:
        lines.append("Accuracy Metrics:")
        for k, v in accuracy_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
    
    if calibration_metrics:
        lines.append("Calibration Metrics:")
        for k, v in calibration_metrics.items():
            lines.append(f"  {k}: {v:.6f}")
    
    if per_class_metrics:
        lines.append("Per-class Metrics:")
        for k, v in per_class_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
    
    if margin_metrics:
        lines.append("Margin Statistics:")
        for k, v in margin_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
    
    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    
    num_samples = 1000
    num_classes = 10
    
    # Simulate predictions and targets
    targets = np.random.randint(0, num_classes, num_samples)
    logits = np.random.randn(num_samples, num_classes)
    
    # Add some bias to make correct class more likely
    for i in range(num_samples):
        logits[i, targets[i]] += 1.0
    
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(predictions, targets, probabilities)
    
    # Print formatted metrics
    print(format_metrics_for_logging(metrics, "Test"))
    
    # Test margin statistics
    tau = np.random.uniform(0.5, 2.0, num_classes)
    margin_stats = calculator.calculate_margin_statistics(logits, targets, tau)
    print("\nMargin Statistics:")
    for k, v in margin_stats.items():
        print(f"  {k}: {v:.4f}")
