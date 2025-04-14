# modules/mpc/linear.py

from mpyc.runtime import mpc
from utils.constant import DEFAULT_EPOCHS, DEFAULT_LR

class SecureLinearRegression:
    def __init__(self, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR):
        self.epochs = epochs
        self.lr = lr
        self.theta = None  # Model parameters
        self.secfx = mpc.SecFxp()

    async def fit(self, X_parts, y_parts):
        """Securely train linear regression using gradient descent.

        Args:
            X_parts (List[List[List[secfx]]]): List of X matrices from parties (all combined).
            y_parts (List[List[secfx]]): List of y vectors from parties (all combined).
        """
        
        # Concatenate data from all parties (already flattened)
        X = X_parts[0]  # shape: (n_samples, n_features)
        y = y_parts[0]  # shape: (n_samples,)
        n_samples = len(y)
        n_features = len(X[0])

        print(f"[Party {mpc.pid}] ✅ Loaded {n_samples} samples, {n_features} features")
        
        # Initialize theta (model weights) to zeros
        theta = [self.secfx(0) for _ in range(n_features)]
        lr_sec = self.secfx(self.lr)

        print(f"\n[Party {mpc.pid}] 🔎 Start learning with {self.epochs} iterations and learning rate {self.lr}")
        for epoch in range(self.epochs):
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
            theta = [theta[j] - lr_sec * gradients[j] for j in range(n_features)]
            
            # Logging: Print theta every 10 iterations
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                theta_debug = await mpc.output(theta)
                print(f"[Party {mpc.pid}] 🧮 Epoch {epoch + 1}: theta = {[float(t) for t in theta_debug]}")

        # Reveal model weights to all parties
        print(f"\n[Party {mpc.pid}] ⌛ Reaching final training epoch...")
        try:
            theta_open = await mpc.output(theta)
            self.theta = [float(t) for t in theta_open]
            print(f"[Party {mpc.pid}] ✅ Training complete. Model weights: {self.theta}")
        except Exception as e:
            print(f"[Party {mpc.pid}] ❗ ERROR during mpc.output: {e}")
            self.theta = []

    async def predict(self, X_input):
        """Securely predict using the trained model.

        Args:
            X_input (List[List[secfx]]): New input data (securely shared, same format).

        Returns:
            List[float]: Predicted values.
        """
        if self.theta is None:
            raise ValueError("Model not trained. Call fit() before predict().")

        theta_sec = [self.secfx(t) for t in self.theta]
        predictions = [sum(x_i[j] * theta_sec[j] for j in range(len(theta_sec))) for x_i in X_input]

        try:
            preds_open = await mpc.output(predictions)
            return [float(p) for p in preds_open]
        except Exception as e:
            print(f"[Party {mpc.pid}] ❗ ERROR during prediction output: {e}")
            return []
