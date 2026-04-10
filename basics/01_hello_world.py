# Lesson 1: Simulating Model Drift and Performance Decay
# Topic: Why MLOps Retraining Schedules Fail — Models Don't Forget, They Get Shocked
# Level: Basics
# What you'll learn:
#   - How to simulate a trained ML model and production data
#   - Visualize model performance over time
#   - Understand the difference between gradual decay and sudden shocks
#   - See why calendar-based retraining doesn't align with actual drift
# Run: python basics/01_hello_world.py
# Time: ~10 min

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def generate_training_data(n_samples: int = 1000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for a binary classification problem.
    
    This simulates a fraud detection scenario where features follow
    a normal distribution and the target is based on a threshold.
    
    Args:
        n_samples: Number of training samples
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(random_state)
    
    # Feature: transaction amount (mean=100, std=30)
    X = np.random.normal(loc=100, scale=30, size=(n_samples, 1))
    
    # Label: fraud if amount > 130 (with some noise)
    y = (X[:, 0] > 130).astype(int)
    noise = np.random.binomial(1, 0.1, n_samples)  # 10% label noise
    y = np.abs(y - noise)  # Flip some labels
    
    return X, y


def simple_model_predict(X: np.ndarray, threshold: float = 130.0) -> np.ndarray:
    """
    A simple threshold-based classifier (simulating a trained model).
    
    In reality, this would be a trained sklearn model, but we use
    a simple rule to make the concept crystal clear.
    
    Args:
        X: Feature array
        threshold: Decision boundary
        
    Returns:
        Binary predictions
    """
    return (X[:, 0] > threshold).astype(int)


def generate_production_data_with_shock(
    n_days: int = 90,
    samples_per_day: int = 100,
    shock_day: int = 45,
    shock_magnitude: float = 20.0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate production data that experiences a distribution shock.
    
    This simulates what happens in real production: data distribution
    is stable, then suddenly shifts (e.g., new fraud patterns emerge).
    
    Args:
        n_days: Number of days to simulate
        samples_per_day: Samples per day
        shock_day: Day when distribution shifts
        shock_magnitude: How much the mean shifts
        
    Returns:
        List of (features, labels) for each day
    """
    daily_data = []
    
    for day in range(n_days):
        # Before shock: mean=100, after shock: mean=120
        if day < shock_day:
            mean = 100.0
        else:
            mean = 100.0 + shock_magnitude
        
        X = np.random.normal(loc=mean, scale=30, size=(samples_per_day, 1))
        y = (X[:, 0] > 130).astype(int)
        noise = np.random.binomial(1, 0.1, samples_per_day)
        y = np.abs(y - noise)
        
        daily_data.append((X, y))
    
    return daily_data


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


def main():
    """
    Main demonstration: Show how model performance degrades due to
    distribution shock, not gradual forgetting.
    """
    print("=" * 60)
    print("Lesson 1: Model Drift Simulation")
    print("=" * 60)
    
    # Step 1: Train a simple model on historical data
    print("\n[Step 1] Training model on historical data...")
    X_train, y_train = generate_training_data(n_samples=1000)
    train_accuracy = calculate_accuracy(y_train, simple_model_predict(X_train))
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Training data mean: {X_train.mean():.2f}")
    
    # Step 2: Simulate production data over 90 days with a shock at day 45
    print("\n[Step 2] Simulating 90 days of production data...")
    print("Distribution shock occurs at day 45 (mean shifts from 100 → 120)")
    
    daily_data = generate_production_data_with_shock(
        n_days=90,
        samples_per_day=100,
        shock_day=45,
        shock_magnitude=20.0
    )
    
    # Step 3: Calculate daily accuracy
    daily_accuracies = []
    for day, (X_day, y_day) in enumerate(daily_data):
        y_pred = simple_model_predict(X_day)
        acc = calculate_accuracy(y_day, y_pred)
        daily_accuracies.append(acc)
    
    # Step 4: Visualize the results
    print("\n[Step 3] Plotting performance over time...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Model accuracy over time
    days = list(range(len(daily_accuracies)))
    ax1.plot(days, daily_accuracies, linewidth=2, color='#2E86AB')
    ax1.axvline(x=45, color='red', linestyle='--', linewidth=2, label='Distribution Shock')
    ax1.axhline(y=train_accuracy, color='green', linestyle=':', linewidth=2, label='Training Accuracy')
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Performance Over Time (Shock-Based Degradation)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution shift visualization
    pre_shock_means = [daily_data[i][0].mean() for i in range(45)]
    post_shock_means = [daily_data[i][0].mean() for i in range(45, 90)]
    
    ax2.plot(range(45), pre_shock_means, linewidth=2, color='blue', label='Pre-shock')
    ax2.plot(range(45, 90), post_shock_means, linewidth=2, color='orange', label='Post-shock')
    ax2.axvline(x=45, color='red', linestyle='--', linewidth=2, label='Shock Event')
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Feature Mean', fontsize=12)
    ax2.set_title('Data Distribution Shift', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_model_drift_simulation.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as '01_model_drift_simulation.png'")
    plt.show()
    
    # Step 5: Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print(f"Pre-shock accuracy (days 0-44): {np.mean(daily_accuracies[:45]):.3f}")
    print(f"Post-shock accuracy (days 45-89): {np.mean(daily_accuracies[45:]):.3f}")
    print(f"Performance drop: {(np.mean(daily_accuracies[:45]) - np.mean(daily_accuracies[45:])):.3f}")
    print("\n→ Performance degraded SUDDENLY at day 45, not gradually")
    print("→ Calendar-based retraining (e.g., monthly) would miss this")
    print("→ We need SHOCK-BASED retraining triggered by distribution changes")


if __name__ == "__main__":
    main()

# ─── CHALLENGE ──────────────────────────────────────
# Try: Modify shock_day to 30 and shock_magnitude to 30.0
#      How does this affect the performance drop?
# Hint: Larger shocks = larger performance drops
# Next: basics/02_core_concepts.py (learn statistical drift detection)
# ─────────────────────────────────────────────────────