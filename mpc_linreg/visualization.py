import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import math

def plot_actual_vs_predicted(X, y, theta):
    """Plot actual vs predicted target values and show evaluation metrics."""

    def flatten_row(row):
        while isinstance(row, list):
            if len(row) == 0:
                return 0.0  # or raise an error
            row = row[0]
        return row

    weights = [float(w) for w in theta[:-1]]
    bias = float(theta[-1])

    X_no_bias = [
        [float(flatten_row(val)) for val in row[:-1]]
        for row in X
    ]

    y_true = [float(flatten_row(val)) for val in y]

    y_pred = [
        sum(w * x for w, x in zip(weights, row)) + bias
        for row in X_no_bias
    ]

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Plotting
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
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
