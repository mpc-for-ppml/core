# utils/visualization.py

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_curve, roc_auc_score
import math

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual vs predicted target values and show RMSE & R² Score.

    Args:
        X: Feature matrix (list of lists, each includes bias term).
        y: True target values (list).
        theta: Trained weights (bias is last element).
    """

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Plot: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal")

    # Metrics text
    metrics_text = f"RMSE: {rmse:.3f}\nR² Score: {r2:.3f}"
    plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted (Linear Regression)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_logistic_evaluation_report(X, y_true, theta, approx_sigmoid_fn, mpc):
    """
    Evaluate and visualize logistic regression results.

    Args:
        X: Feature matrix (already includes bias term).
        y_true: True binary labels.
        theta: Model weights (last element is bias).
        approx_sigmoid_fn: Function to apply approximate sigmoid.
        mpc: MPyC runtime object (used for awaiting outputs).
    """
    async def evaluate():
        # Predict using sigmoid(dot(x, theta))
        sigmoid_outputs = []
        for x in X:
            dot = sum(a * b for a, b in zip(x, theta))
            sigmoid = approx_sigmoid_fn(dot)
            sigmoid_outputs.append(await mpc.output(sigmoid))

        # Convert sigmoid probabilities to binary predictions
        y_pred = [1 if p >= 0.5 else 0 for p in sigmoid_outputs]

        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        print(f"\n[Party {mpc.pid}] 📊 Showing the evaluation report...")
        print(report)

        # ROC-AUC Curve (only on Party 0)
        if mpc.pid == 0:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)

            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("AUC-ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return evaluate()
