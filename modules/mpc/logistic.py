# modules/mpc/logistic.py

from mpyc.runtime import mpc

async def secure_logistic_regression(X_parts, y_parts, epochs=200, lr=0.2):
    """Secure logistic regression using gradient descent in MPyC.

    Args:
        X_parts (List[List[List[secfx]]]): List of X matrices from parties (all combined).
        y_parts (List[List[secfx]]): List of y vectors from parties (all combined).
        epochs (int): Number of gradient descent iterations.
        lr (float): Learning rate.

    Returns:
        List[float]: Final model weights (theta), revealed to all parties.
    """
    secfx = mpc.SecFxp()

    # Concatenate data
    X = X_parts[0]
    y = y_parts[0]
    n_samples = len(y)
    n_features = len(X[0])

    print(X[0], X[-1])
    print(y[0], y[-1])
    print(f"[Party {mpc.pid}] ✅ Loaded {n_samples} samples, {n_features} features")

    # Initialize model weights
    theta = [secfx(0) for _ in range(n_features)]

    print(f"\n[Party {mpc.pid}] 🔎 Start logistic regression with {epochs} iterations and learning rate {lr}")

    def sigmoid(x):
        # 5th-order Taylor approx: sigmoid(x) ≈ 0.5 + 0.25x - x³/48 + x⁵/480
        const_05 = secfx(0.5)
        const_025 = secfx(0.25)
        x3 = x * x * x
        x5 = x3 * x * x
        return const_05 + const_025 * x - (x3 / 48) + (x5 / 480)

    for epoch in range(epochs):
        # Compute predictions: sigmoid(X @ theta)
        y_pred = [sigmoid(sum(x_i[j] * theta[j] for j in range(n_features))) for x_i in X]

        # Compute error: y_pred - y
        error = [y_pred[i] - y[i] for i in range(n_samples)]

        # Compute gradients
        gradients = []
        for j in range(n_features):
            grad_j = sum(error[i] * X[i][j] for i in range(n_samples)) / n_samples
            gradients.append(grad_j)

        # Update weights
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
