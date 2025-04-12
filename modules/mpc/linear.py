# modules/mpc/linear.py

import numpy as np
from mpyc.runtime import mpc

secfxp = mpc.SecFxp()

async def secure_linear_regression(X_parties, y_parties):
    """Performs secure linear regression using MPC."""
    # Step 0: Flatten inputs from all parties
    X = sum(X_parties, [])
    y = sum(y_parties, [])
    
    # Check for data size mismatch
    m, n = len(X), len(X[0])
    assert len(X) == len(y), f"Mismatch: X has {m} samples, but y has {len(y)} labels"

    print(f"[Party {mpc.pid}] ✅ Loaded {m} samples, {n} features")
    
    # Step 1: Convert to secret-shared values (local data)
    X_sec = [[secfxp(xij) for xij in xi] for xi in X]
    y_sec = [secfxp(yi) for yi in y]
    
    # Step 2: Compute XtX = X^T * X and Xty = X^T * y securely
    Xt = list(zip(*X_sec))
    XtX = [[sum(xi * xj for xi, xj in zip(row_i, row_j)) for row_j in Xt] for row_i in Xt]
    Xty = [sum(xi * yi for xi, yi in zip(row, y_sec)) for row in Xt]

    # Step 3: Open XtX and Xty result to all parties
    XtX_flat = [elem for row in XtX for elem in row]
    XtX_open_flat = await mpc.output(XtX_flat)
    n = len(XtX)
    XtX_open = [XtX_open_flat[i * n:(i + 1) * n] for i in range(n)]

    Xty_open = await mpc.output(Xty)

    # Step 4: Solve for theta (plain domain here, for simplicity)
    XtX_np = np.array([[float(x) for x in row] for row in XtX_open])
    Xty_np = np.array([float(v) for v in Xty_open])
    lambda_reg = 1e-4
    XtX_np += lambda_reg * np.identity(XtX_np.shape[0])

    # Solve theta using numpy (insecure, but after open)
    theta = np.linalg.solve(XtX_np, Xty_np)
    return theta
