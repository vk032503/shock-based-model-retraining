# Lesson 2: Statistical Drift Detection Fundamentals
# Topic: Why MLOps Retraining Schedules Fail — Models Don't Forget, They Get Shocked
# Level: Basics
# What you'll learn:
#   - Compare two distributions statistically (not just visually)
#   - Implement the Kolmogorov-Smirnov (KS) test for drift detection
#   - Understand p-values and statistical significance
#   - Detect when a distribution has actually shifted vs. normal variation
# Run: python basics/02_core_concepts.py
# Time: ~15 min

import numpy as np
from scipy import stats
from typing import Tuple
import matplotlib.pyplot as plt


def kolmogorov_smirnov_test(
    reference_data: np.ndarray,
    current_data: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, float, bool]:
    """
    Perform Kolmogorov-Smirnov test to detect distribution drift.
    
    The KS test compares two distributions by measuring the maximum
    distance between their cumulative distribution functions (CDFs).
    
    Why KS test?
    - Non-parametric (no assumptions about distribution shape)
    - Sensitive to both location and shape changes
    - Widely used in production ML systems
    
    Args:
        reference_data: Training/baseline distribution
        current_data: Production/current distribution
        alpha: Significance level (default 0.05 = 95% confidence)
        
    Returns:
        Tuple of (KS statistic, p-value, is_drift_detected)
    """
    # Perform two-sample KS test
    ks_statistic, p_value = stats.ks_2samp(reference_data, current_data)
    
    # Drift detected if p-value < alpha
    # Low p-value means distributions are significantly different
    is_drift = p_value < alpha
    
    return ks_statistic, p_value, is_drift


def visualize_distributions(
    reference_data: np.ndarray,
    current_data: np.ndarray,
    ks_stat: float,
    p_value: float,
    title: str = "Distribution Comparison"
) -> None:
    """
    Visualize two distributions and their CDFs to understand drift.
    
    Args:
        reference_data: Baseline distribution
        current_data: Current distribution
        ks_stat: KS test statistic
        p_value: KS test p-value
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histograms (PDFs)
    ax1.hist(reference_data, bins=30, alpha=0.6, label='Reference', color='blue', density=True)
    ax1.hist(current_data, bins=30, alpha=0.6, label='Current', color='orange', density=True)
    ax1.set_xlabel('Feature Value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Probability Distributions (PDFs)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative distributions (CDFs)
    # The KS statistic is the maximum vertical distance between these curves
    sorted_ref = np.sort(reference_data)
    sorted_cur = np.sort(current_data)
    
    cdf_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)
    cdf_cur = np.arange(1, len(sorted_cur) + 1) / len(sorted_cur)
    
    ax2.plot(sorted_ref, cdf_ref, label='Reference CDF', color='blue', linewidth=2)
    ax2.plot(sorted_cur, cdf_cur, label='Current CDF', color='orange', linewidth=2)
    ax2.set_xlabel('Feature Value', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title(f'CDFs (KS stat={ks_stat:.3f}, p={p_value:.4f})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('02_distribution_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def demonstrate_no_drift():
    """
    Scenario 1: No drift - both samples from same distribution.
    
    This shows what normal variation looks like. Even samples from
    the same distribution will have small differences.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 1: No Drift (Normal Variation)")
    print("=" * 60)
    
    # Both samples from N(100, 30)
    np.random.seed(42)
    reference = np.random.normal(loc=100, scale=30, size=1000)
    current = np.random.normal(loc=100, scale=30, size=1000)
    
    ks_stat, p_value, is_drift = kolmogorov_smirnov_test(reference, current)
    
    print(f"Reference mean: {reference.mean():.2f}")
    print(f"Current mean: {current.mean():.2f}")
    print(f"\nKS Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Drift detected: {is_drift}")
    print("\nInterpretation:")
    print(f"→ P-value {p_value:.4f} > 0.05 means we CANNOT reject null hypothesis")
    print("→ Distributions are statistically similar (no drift)")
    
    visualize_distributions(reference, current, ks_stat, p_value, "No Drift Scenario")


def demonstrate_drift():
    """
    Scenario 2: Clear drift - distribution shift detected.
    
    This simulates what happens in production when data distribution
    changes (e.g., new fraud patterns, market shift, user behavior change).
    """
    print("\n" + "=" * 60)
    print("SCENARIO 2: Distribution Shock (Drift Detected)")
    print("=" * 60)
    
    # Reference: N(100, 30), Current: N(120, 30) - mean shifted by 20
    np.random.seed(42)
    reference = np.random.normal(loc=100, scale=30, size=1000)
    current = np.random.normal(loc=120, scale=30, size=1000)  # SHOCK!
    
    ks_stat, p_value, is_drift = kolmogorov_smirnov_test(reference, current)
    
    print(f"Reference mean: {reference.mean():.2f}")
    print(f"Current mean: {current.mean():.2f}")
    print(f"Mean shift: {current.mean() - reference.mean():.2f}")
    print(f"\nKS Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Drift detected: {is_drift}")
    print("\nInterpretation:")
    print(f"→ P-value {p_value:.6f} < 0.05 means we REJECT null hypothesis")
    print("→ Distributions are statistically different (DRIFT DETECTED)")
    print("→ This should trigger model retraining!")
    
    visualize_distributions(reference, current, ks_stat, p_value, "Drift Detected Scenario")


def demonstrate_subtle_drift():
    """
    Scenario 3: Subtle drift - small but statistically significant shift.
    
    This shows the power of statistical tests: they can detect shifts
    that might not be obvious visually but are real.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 3: Subtle Drift (Small but Significant)")
    print("=" * 60)
    
    # Reference: N(100, 30), Current: N(105, 30) - small mean shift
    np.random.seed(42)
    reference = np.random.normal(loc=100, scale=30, size=1000)
    current = np.random.normal(loc=105, scale=30, size=1000)  # Small shift
    
    ks_stat, p_value, is_drift = kolmogorov_smirnov_test(reference, current)
    
    print(f"Reference mean: {reference.mean():.2f}")
    print(f"Current mean: {current.mean():.2f}")
    print(f"Mean shift: {current.mean() - reference.mean():.2f}")
    print(f"\nKS Statistic: {ks_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Drift detected: {is_drift}")
    print("\nInterpretation:")
    print("→ Small shift but still statistically significant")
    print("→ Might not need immediate retraining, but worth monitoring")
    print("→ This is where adaptive thresholds help (covered in advanced lessons)")


def main():
    """
    Run all drift detection scenarios to understand statistical testing.
    """
    print("=" * 60)
    print("Lesson 2: Statistical Drift Detection")
    print("=" * 60)
    print("\nWe'll test 3 scenarios using the Kolmogorov-Smirnov test:")
    print("1. No drift (normal variation)")
    print("2. Clear drift (distribution shock)")
    print("3. Subtle drift (small but significant)")
    
    demonstrate_no_drift()
    demonstrate_drift()
    demonstrate_subtle_drift()
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("✓ KS test detects distribution changes statistically")
    print("✓ P-value < 0.05 → drift detected (95% confidence)")
    print("✓ Works for any distribution shape (non-parametric)")
    print("✓ More reliable than visual inspection or simple mean comparison")
    print("\nPlots saved as '02_distribution_comparison.png'")


if __name__ == "__main__":
    main()

# ─── CHALLENGE ──────────────────────────────────────
# Try: Change alpha to 0.01 (99% confidence) in scenario 3
#      Does subtle drift still get detected?
# Hint: Higher confidence = stricter threshold = fewer detections
# Next: basics/03_first_real_program.py (build a drift detector)
# ─────────────────────────────────────────────────────