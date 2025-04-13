# modules/mpc/logistic.py

from mpyc.runtime import mpc

# Default values — same as your function signature
DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.01

secfxp = mpc.SecFxp()

def approx_log(x, terms=5):
    one = secfxp(1)
    x_minus_1 = x - one
    result = secfxp(0)
    sign = 1
    power = x_minus_1
    for n in range(1, terms + 1):
        term = power / n
        result += sign * term
        power *= x_minus_1
        sign *= -1
    return result

def approx_sigmoid(x):
    # 5th-order Taylor approx: sigmoid(x) ≈ 0.5 + 0.25x - x³/48 + x⁵/480
    const_05 = secfxp(0.5)
    const_025 = secfxp(0.25)
    x3 = x * x * x
    x5 = x3 * x * x
    return const_05 + const_025 * x - (x3 / 48) + (x5 / 480)

async def secure_logistic_regression(X_parts, y_parts, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
    """Secure logistic regression using gradient descent in MPyC.

    Args:
        X_parts (List[List[List[secfx]]]): List of X matrices from parties (all combined).
        y_parts (List[List[secfx]]): List of y vectors from parties (all combined).
        epochs (int): Number of gradient descent iterations.
        lr (float): Learning rate.

    Returns:
        List[float]: Final model weights (theta), revealed to all parties.
    """
    # Concatenate data
    X = X_parts[0]
    y = y_parts[0]
    n_samples = len(y)
    n_features = len(X[0])

    print(f"[Party {mpc.pid}] ✅ Loaded {n_samples} samples, {n_features} features")

    # Initialize model weights to zero
    theta = [secfxp(0) for _ in range(n_features)]
    lr_sec = secfxp(lr)

    print(f"\n[Party {mpc.pid}] 🔎 Start logistic regression with {epochs} iterations and learning rate {lr}")
    
    # bias term
    bias = secfxp(0)

    for epoch in range(epochs):
        # Compute predictions: sigmoid(X @ theta)
        y_pred = [approx_sigmoid(sum(x_i[j] * theta[j] for j in range(n_features)) + bias) for x_i in X]

        # Compute error: y_pred - y
        error = [y_pred[i] - y[i] for i in range(n_samples)]

        # Compute gradients
        gradients = []
        for j in range(n_features):
            grad_j = sum(error[i] * X[i][j] for i in range(n_samples)) / n_samples
            gradients.append(grad_j)

        # Compute gradient for bias
        grad_bias = sum(error) / n_samples

        # Update theta and bias
        theta = [theta[j] - lr_sec * gradients[j] for j in range(n_features)]
        bias = bias - lr_sec * grad_bias

        # Debug: Print theta every 10 iterations
        if epoch % 10 == 0 or epoch == epochs - 1:
            theta_debug = await mpc.output(theta + [bias])  # Combine for debugging
            epsilon = secfxp(1e-3)
            y_pred_clamped = [mpc.max(epsilon, mpc.min(1 - epsilon, yp)) for yp in y_pred]
            loss_terms = [
                y[i] * approx_log(y_pred_clamped[i]) + (1 - y[i]) * approx_log(1 - y_pred_clamped[i])
                for i in range(n_samples)
            ]
            loss = -sum(loss_terms) / n_samples
            loss_val = await mpc.output(loss)
            print(f"[Party {mpc.pid}] 🧮 Epoch {epoch + 1}: theta = {[float(t) for t in theta_debug]} | loss = {loss_val}")

    # Reveal final model weights
    print(f"[Party {mpc.pid}] ⌛ Reaching final output...")
    try:
        theta_open = await mpc.output(theta + [bias])
        return [float(t) for t in theta_open]
    except Exception as e:
        print(f"[Party {mpc.pid}] ❗ ERROR during mpc.output: {e}")
        return []
