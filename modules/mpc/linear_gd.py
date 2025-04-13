# modules/mpc/linear_gd.py

from mpyc.runtime import mpc

# Default values — same as your function signature
DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.2

async def secure_linear_regression(X_parts, y_parts, epochs=200, lr=0.2):
    """Secure multilinear regression using gradient descent in MPyC.

    Args:
        X_parts (List[List[List[secfx]]]): List of X matrices from parties (all combined).
        y_parts (List[List[secfx]]): List of y vectors from parties (all combined).
        epochs (int): Number of gradient descent iterations.
        lr (float): Learning rate.

    Returns:
        List[secfx]: Model parameters (theta).
    """
    secfx = mpc.SecFxp()  # secure fixed-point type

    # Concatenate data from all parties (already flattened)
    X = X_parts[0]  # shape: (n_samples, n_features)
    y = y_parts[0]  # shape: (n_samples,)
    n_samples = len(y)
    n_features = len(X[0])
    
    print(f"[Party {mpc.pid}] ✅ Loaded {n_samples} samples, {n_features} features")

    # Initialize theta (model weights) to zeros
    theta = [secfx(0) for _ in range(n_features)]

    print(f"\n[Party {mpc.pid}] 🔎 Start learning with {epochs} iterations and learning rate {lr}")
    for epoch in range(epochs):
        # Compute predictions: y_pred = X @ theta
        y_pred = [sum(x_i[j] * theta[j] for j in range(n_features)) for x_i in X]

        # Compute error = y_pred - y
        error = [y_pred[i] - y[i] for i in range(n_samples)]

        # Compute gradients
        gradients = []
        for j in range(n_features):
            grad_j = sum(error[i] * X[i][j] for i in range(n_samples)) / n_samples
            gradients.append(grad_j)

        # Update theta
        theta = [theta[j] - secfx(lr) * gradients[j] for j in range(n_features)]
        
        # Iteration logging
        print(f"[Party {mpc.pid}] 🔄 Iteration {epoch}")

    # Reveal model weights to all parties
    print(f"[Party {mpc.pid}] ⌛ Reaching final output...")
    try:
        theta_open = await mpc.output(theta)
        return [float(t) for t in theta_open]
    except Exception as e:
        print(f"[Party {mpc.pid}] ❗ ERROR during mpc.output: {e}")
        return []
