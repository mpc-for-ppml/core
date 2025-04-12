from mpyc.runtime import mpc

secfxp = mpc.SecFxp()

async def sigmoid(x):
    """Polynomial approximation of the sigmoid function."""
    const_05 = secfxp(0.5)
    const_0197 = secfxp(0.197)
    const_0004 = secfxp(0.004)

    x2 = x * x
    x3 = x2 * x
    return const_05 + const_0197 * x - const_0004 * x3

async def secure_logistic_regression(X_parties, y_parties, num_iters=20, lr=0.05):
    """Secure logistic regression using secret-shared gradient descent."""
    # Step 0: Flatten inputs from all parties
    X = sum(X_parties, [])
    y = sum(y_parties, [])

    # Step 1: Check for data size mismatch
    m, n = len(X), len(X[0])
    assert len(X) == len(y), f"Mismatch: X has {m} samples, but y has {len(y)} labels"

    print(f"[Party {mpc.pid}] ✅ Loaded {m} samples, {n} features")

    # Step 2: Convert to secret-shared values (local data)
    X_sec = [[secfxp(xij) for xij in xi] for xi in X]
    y_sec = [secfxp(yi) for yi in y]
    theta = [secfxp(0) for _ in range(n)]

    # Step 3: Learning
    print(f"\n[Party {mpc.pid}] 🔎 Start learning with {num_iters} iterations and learning rate {lr}")
    for iteration in range(num_iters):
        # Compute X * theta
        preds = []
        for xi in X_sec:
            dot = sum(xij * tj for xij, tj in zip(xi, theta))
            sig = await sigmoid(dot)
            preds.append(sig)

        # Compute gradient: grad = (1/m) * X^T * (preds - y)
        errors = [pi - yi for pi, yi in zip(preds, y_sec)]
        grad = []
        
        for j in range(n):
            col_j = [xij[j] for xij in X_sec]
            g_j = sum(xj * err for xj, err in zip(col_j, errors)) / m
            grad.append(g_j)

        # Update theta
        theta = [t - lr * g for t, g in zip(theta, grad)]

        # Iteration logging
        print(f"[Party {mpc.pid}] 🔄 Iteration {iteration}")

    # Step 4: Reveal final model
    print(f"[Party {mpc.pid}] ⌛ Reaching final output...")
    try:
        theta_open = await mpc.output(theta)
        return [float(t) for t in theta_open]
    except Exception as e:
        print(f"[Party {mpc.pid}] ❗ ERROR during mpc.output: {e}")
        return []
